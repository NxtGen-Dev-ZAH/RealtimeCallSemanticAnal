#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-First Diarization Approach
Uses FLAN-T5 to analyze transcription first, then uses that information for diarization.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend/src and backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir / "src"))
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
from call_analysis.preprocessing import AudioProcessor
from config import Config
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load environment variables
load_dotenv(dotenv_path=backend_dir / ".env")


def analyze_transcription_with_llm(transcription_text: str, llm_model, llm_tokenizer):
    """
    Use FLAN-T5 to analyze transcription and identify speaker roles.
    
    Args:
        transcription_text: Full transcription text
        llm_model: Loaded FLAN-T5 model
        llm_tokenizer: FLAN-T5 tokenizer
        
    Returns:
        Dictionary with speaker role predictions
    """
    print("🤖 Step 1: Analyzing transcription with FLAN-T5...")
    print()
    
    # Split transcription into segments (by sentences or paragraphs)
    sentences = transcription_text.split('. ')
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        if len(current_segment) + len(sentence) < 500:  # Keep segments under 500 chars
            current_segment += sentence + ". "
        else:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence + ". "
    
    if current_segment:
        segments.append(current_segment.strip())
    
    print(f"   Split transcription into {len(segments)} segments for analysis")
    print()
    
    # Analyze each segment to identify if it's agent or customer
    role_predictions = []
    
    for i, segment in enumerate(segments[:50]):  # Limit to first 50 segments for speed
        if len(segment.strip()) < 20:  # Skip very short segments
            continue
        
        try:
            # Create prompt for FLAN-T5
            prompt = f"Classify this call center conversation segment as 'agent' or 'customer'. Segment: {segment[:400]} Classification:"
            
            # Tokenize and generate
            inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Move to device
            if Config.LLM_DEVICE == 'cuda' and torch.cuda.is_available():
                device = next(llm_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode response
            response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Determine role
            is_agent = 'agent' in response and 'customer' not in response
            is_customer = 'customer' in response and 'agent' not in response
            
            if is_agent:
                role = "AGENT"
            elif is_customer:
                role = "CUSTOMER"
            else:
                # Use heuristics if unclear
                agent_keywords = ['thank you for calling', 'how can i help', 'policy', 'coverage', 'premium', 'i can', 'i will']
                customer_keywords = ['i want', 'i need', 'my', 'i have', 'yes', 'no']
                
                segment_lower = segment.lower()
                agent_score = sum(1 for kw in agent_keywords if kw in segment_lower)
                customer_score = sum(1 for kw in customer_keywords if kw in segment_lower)
                
                role = "AGENT" if agent_score > customer_score else "CUSTOMER"
            
            role_predictions.append({
                'segment_index': i,
                'text': segment[:200],  # Truncate for display
                'predicted_role': role,
                'llm_response': response
            })
            
            if (i + 1) % 10 == 0:
                print(f"   Analyzed {i + 1}/{min(50, len(segments))} segments...")
                
        except Exception as e:
            print(f"   Warning: Failed to analyze segment {i}: {e}")
            continue
    
    print(f"✅ Analyzed {len(role_predictions)} segments")
    print()
    
    # Count role distribution
    agent_count = sum(1 for p in role_predictions if p['predicted_role'] == 'AGENT')
    customer_count = sum(1 for p in role_predictions if p['predicted_role'] == 'CUSTOMER')
    
    print(f"📊 LLM Role Predictions:")
    print(f"   AGENT segments: {agent_count}")
    print(f"   CUSTOMER segments: {customer_count}")
    print()
    
    return role_predictions


def run_llm_first_diarization(
    audio_path: str = None,
    transcription_file: str = None,
    output_dir: str = None,
    clustering_threshold: float = 0.25,
    max_speakers: int = 3
):
    """
    Run diarization using LLM-first approach.
    
    Args:
        audio_path: Path to audio file
        transcription_file: Path to existing transcription file (optional)
        output_dir: Directory to save output
        clustering_threshold: Clustering threshold for diarization
        max_speakers: Maximum number of speakers
    """
    print("=" * 70)
    print("LLM-FIRST DIARIZATION APPROACH")
    print("=" * 70)
    print()
    
    # Find audio file if not provided
    if audio_path is None:
        print("🔍 Searching for audio file...")
        try:
            from run_diarization_with_llm import find_audio_file
            audio_path = find_audio_file("real_audio.wav")
            print(f"✅ Found audio file: {audio_path}")
        except:
            print("❌ Error: Audio file not found")
            sys.exit(1)
    else:
        if not os.path.exists(audio_path):
            print(f"❌ Error: Audio file not found: {audio_path}")
            sys.exit(1)
        audio_path = os.path.abspath(audio_path)
    
    # Load or create transcription
    transcription_text = ""
    if transcription_file and os.path.exists(transcription_file):
        print(f"📄 Loading existing transcription: {transcription_file}")
        with open(transcription_file, 'r', encoding='utf-8') as f:
            if transcription_file.endswith('.json'):
                data = json.load(f)
                transcription_text = data.get('text', '')
            else:
                transcription_text = f.read()
        print(f"✅ Loaded {len(transcription_text)} characters")
        print()
    else:
        print("📝 Transcribing audio first...")
        print()
        # Initialize AudioProcessor for transcription
        audio_processor = AudioProcessor(
            model_size=Config.WHISPER_MODEL_SIZE or "base",
            hf_token=Config.HF_TOKEN
        )
        call_id_temp = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        transcription = audio_processor.transcribe_audio(audio_path, call_id_temp)
        transcription_text = transcription.get('text', '')
        print(f"✅ Transcribed {len(transcription_text)} characters")
        print()
    
    # Load FLAN-T5 model
    print("🤖 Loading FLAN-T5 model...")
    try:
        llm_tokenizer = T5Tokenizer.from_pretrained(Config.LLM_ROLE_IDENTIFICATION_MODEL)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        llm_model = T5ForConditionalGeneration.from_pretrained(Config.LLM_ROLE_IDENTIFICATION_MODEL)
        
        if Config.LLM_DEVICE == 'cuda' and torch.cuda.is_available():
            llm_model = llm_model.to('cuda')
        else:
            llm_model = llm_model.to('cpu')
        
        llm_model.eval()
        print("✅ FLAN-T5 model loaded")
        print()
    except Exception as e:
        print(f"❌ Error loading FLAN-T5: {e}")
        sys.exit(1)
    
    # Step 1: Analyze transcription with LLM
    role_predictions = analyze_transcription_with_llm(transcription_text, llm_model, llm_tokenizer)
    
    # Step 2: Run diarization with adjusted parameters based on LLM insights
    print("=" * 70)
    print("STEP 2: SPEAKER DIARIZATION")
    print("=" * 70)
    print()
    
    # Adjust clustering threshold based on LLM predictions
    # If LLM found clear agent/customer distinction, we can be more confident
    agent_ratio = sum(1 for p in role_predictions if p['predicted_role'] == 'AGENT') / len(role_predictions) if role_predictions else 0.5
    
    if 0.3 < agent_ratio < 0.7:  # Balanced conversation
        suggested_threshold = 0.2  # Lower threshold to detect 2 speakers
        print(f"💡 LLM suggests balanced conversation (agent ratio: {agent_ratio:.2f})")
        print(f"   Using lower clustering threshold: {suggested_threshold}")
    else:
        suggested_threshold = clustering_threshold
        print(f"💡 Using default clustering threshold: {suggested_threshold}")
    
    print()
    
    # Run diarization
    call_id = f"llm_first_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🎤 Running speaker diarization...")
    print()
    
    audio_processor = AudioProcessor(
        model_size=Config.WHISPER_MODEL_SIZE or "base",
        hf_token=Config.HF_TOKEN,
        use_llm_diarization=True,
        llm_role_model=Config.LLM_ROLE_IDENTIFICATION_MODEL,
        llm_refinement_model=Config.LLM_REFINEMENT_MODEL,
        llm_device=Config.LLM_DEVICE,
        clustering_threshold=suggested_threshold,
        max_speakers=max_speakers
    )
    
    segments = audio_processor.perform_speaker_diarization(audio_path, call_id)
    
    # Step 3: Combine LLM predictions with diarization results
    print()
    print("=" * 70)
    print("STEP 3: COMBINING LLM PREDICTIONS WITH DIARIZATION")
    print("=" * 70)
    print()
    
    # Map LLM predictions to diarization segments by text similarity
    for seg in segments:
        seg_text = seg.get('text', '').strip().lower()
        if not seg_text:
            continue
        
        # Find best matching LLM prediction
        best_match = None
        best_score = 0
        
        for pred in role_predictions:
            pred_text = pred['text'].lower()
            # Simple text overlap score
            common_words = set(seg_text.split()) & set(pred_text.split())
            score = len(common_words) / max(len(seg_text.split()), 1)
            
            if score > best_score and score > 0.3:  # At least 30% overlap
                best_score = score
                best_match = pred
        
        if best_match:
            # Add LLM prediction as additional metadata
            seg['llm_predicted_role'] = best_match['predicted_role']
            seg['llm_confidence'] = best_score
    
    # Save results
    output_file = output_dir / f"{call_id}_llm_first_diarization.json"
    
    result_data = {
        'call_id': call_id,
        'audio_file': audio_path,
        'timestamp': datetime.now().isoformat(),
        'llm_analysis': {
            'segments_analyzed': len(role_predictions),
            'agent_segments': sum(1 for p in role_predictions if p['predicted_role'] == 'AGENT'),
            'customer_segments': sum(1 for p in role_predictions if p['predicted_role'] == 'CUSTOMER'),
            'predictions': role_predictions
        },
        'diarization': {
            'segments_count': len(segments),
            'speakers_found': len(set(s.get('speaker', '') for s in segments)),
            'segments': segments
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    print("✅ LLM-first diarization completed!")
    print()
    
    # Print summary
    unique_speakers = set(s.get('speaker', '') for s in segments)
    print(f"📊 Final Results:")
    print(f"   Speakers detected: {len(unique_speakers)}")
    print(f"   Segments with LLM role prediction: {sum(1 for s in segments if 'llm_predicted_role' in s)}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run diarization using LLM-first approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use LLM on transcription first, then diarize
  python run_llm_first_diarization.py
  
  # Use existing transcription file
  python run_llm_first_diarization.py --transcription transcription_output.txt
  
  # Specify audio and adjust parameters
  python run_llm_first_diarization.py real_audio.wav --clustering-threshold 0.2
        """
    )
    
    parser.add_argument(
        'audio_file',
        nargs='?',
        default=None,
        help='Path to audio file (default: searches for real_audio.wav)'
    )
    
    parser.add_argument(
        '--transcription',
        type=str,
        default=None,
        help='Path to existing transcription file (.txt or .json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: output/)'
    )
    
    parser.add_argument(
        '--clustering-threshold',
        type=float,
        default=0.25,
        help='Clustering threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=3,
        help='Maximum speakers (default: 3)'
    )
    
    args = parser.parse_args()
    
    run_llm_first_diarization(
        audio_path=args.audio_file,
        transcription_file=args.transcription,
        output_dir=args.output_dir,
        clustering_threshold=args.clustering_threshold,
        max_speakers=args.max_speakers
    )

