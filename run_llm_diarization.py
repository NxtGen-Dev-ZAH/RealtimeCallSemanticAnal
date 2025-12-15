#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Based Diarization
Uses FLAN-T5 to perform speaker diarization directly from transcription text.
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

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
from config import Config
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load environment variables
load_dotenv(dotenv_path=backend_dir / ".env")


def split_into_speaker_segments(text: str) -> List[str]:
    """
    Split transcription into potential speaker segments.
    Uses natural breaks like periods, question marks, and conversation patterns.
    """
    # Split by sentence endings
    sentences = re.split(r'([.!?]\s+)', text)
    
    # Recombine sentences with their punctuation
    segments = []
    current_segment = ""
    
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
        else:
            sentence = sentences[i]
        
        # Group sentences into segments (2-4 sentences per segment)
        current_segment += sentence
        
        # Break on natural conversation patterns
        if (len(current_segment) > 100 and 
            (sentence.strip().endswith(('.', '?', '!')) or 
             any(marker in sentence.lower() for marker in ['okay', 'yes', 'no', 'thank you', 'hello']))):
            segments.append(current_segment.strip())
            current_segment = ""
    
    if current_segment.strip():
        segments.append(current_segment.strip())
    
    return segments


def identify_speaker_with_llm(segment: str, context: str, llm_model, llm_tokenizer, device: str) -> Dict:
    """
    Use FLAN-T5 to identify speaker role for a segment.
    
    Args:
        segment: Current text segment
        context: Previous segments for context
        llm_model: FLAN-T5 model
        llm_tokenizer: FLAN-T5 tokenizer
        device: Device to run on
        
    Returns:
        Dictionary with speaker identification results
    """
    # Create comprehensive prompt
    prompt = f"""Analyze this call center conversation and identify the speaker.

Previous context: {context[-300:] if context else "Start of conversation"}

Current segment: {segment[:400]}

Identify:
1. Is this the AGENT or CUSTOMER speaking?
2. Is this a speaker change from the previous segment? (yes/no)

Answer format: Speaker: [AGENT/CUSTOMER], Change: [yes/no]"""
    
    try:
        # Tokenize
        inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=600, truncation=True)
        
        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            device_obj = next(llm_model.parameters()).device
            inputs = {k: v.to(device_obj) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_length=100,
                num_beams=3,
                early_stopping=True,
                do_sample=False,
                temperature=0.3
            )
        
        # Decode
        response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        # Parse response
        speaker = None
        is_change = False
        
        if 'agent' in response:
            speaker = "AGENT"
        elif 'customer' in response:
            speaker = "CUSTOMER"
        
        if 'change: yes' in response or 'change:yes' in response or 'yes' in response.split('change:')[1] if 'change:' in response else False:
            is_change = True
        
        # Fallback heuristics if LLM response unclear
        if speaker is None:
            agent_keywords = ['thank you for calling', 'how can i help', 'policy', 'coverage', 
                            'premium', 'i can', 'i will', 'let me', 'i understand', 'i appreciate']
            customer_keywords = ['i want', 'i need', 'my', 'i have', 'yes', 'no', 'okay', 
                               'i think', 'i believe', 'i would like']
            
            segment_lower = segment.lower()
            agent_score = sum(1 for kw in agent_keywords if kw in segment_lower)
            customer_score = sum(1 for kw in customer_keywords if kw in segment_lower)
            
            speaker = "AGENT" if agent_score > customer_score else "CUSTOMER"
        
        return {
            'speaker': speaker,
            'is_speaker_change': is_change,
            'llm_response': response,
            'confidence': 'high' if speaker and 'agent' in response or 'customer' in response else 'medium'
        }
        
    except Exception as e:
        # Fallback to heuristics
        agent_keywords = ['thank you for calling', 'how can i help', 'policy', 'coverage']
        customer_keywords = ['i want', 'i need', 'my', 'i have']
        
        segment_lower = segment.lower()
        agent_score = sum(1 for kw in agent_keywords if kw in segment_lower)
        customer_score = sum(1 for kw in customer_keywords if kw in segment_lower)
        
        return {
            'speaker': "AGENT" if agent_score > customer_score else "CUSTOMER",
            'is_speaker_change': True,  # Assume change on error
            'llm_response': f"Error: {str(e)}",
            'confidence': 'low'
        }


def perform_llm_diarization(transcription_text: str, llm_model, llm_tokenizer, device: str) -> List[Dict]:
    """
    Perform complete diarization using LLM.
    
    Args:
        transcription_text: Full transcription text
        llm_model: FLAN-T5 model
        llm_tokenizer: FLAN-T5 tokenizer
        device: Device to run on
        
    Returns:
        List of diarization segments with speaker labels
    """
    print("🤖 Performing LLM-based diarization...")
    print()
    
    # Split into segments
    segments = split_into_speaker_segments(transcription_text)
    print(f"   Split into {len(segments)} segments")
    print()
    
    # Process segments
    diarization_segments = []
    previous_speaker = None
    context = ""
    current_time = 0.0
    segment_duration = 3.0  # Estimate 3 seconds per segment (will be adjusted)
    
    print("   Analyzing segments with FLAN-T5...")
    
    for i, segment in enumerate(segments):
        if len(segment.strip()) < 10:  # Skip very short segments
            continue
        
        # Identify speaker
        result = identify_speaker_with_llm(segment, context, llm_model, llm_tokenizer, device)
        
        speaker = result['speaker']
        is_change = result['is_speaker_change']
        
        # Determine if this is a new speaker
        if previous_speaker is None:
            # First segment
            is_change = True
        elif is_change or speaker != previous_speaker:
            # Speaker changed
            is_change = True
        else:
            # Same speaker continues
            is_change = False
        
        # Estimate duration based on text length (rough: 150 words per minute)
        words = len(segment.split())
        estimated_duration = max(1.0, (words / 150) * 60)  # Convert to seconds
        
        # Create segment
        diarization_segments.append({
            'start': current_time,
            'end': current_time + estimated_duration,
            'text': segment.strip(),
            'speaker': speaker,
            'is_speaker_change': is_change,
            'llm_confidence': result['confidence'],
            'segment_index': i
        })
        
        # Update for next iteration
        current_time += estimated_duration
        previous_speaker = speaker
        context += segment + " "
        
        # Keep context manageable
        if len(context) > 1000:
            context = context[-500:]
        
        # Progress update
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(segments)} segments...")
    
    print(f"✅ Analyzed {len(diarization_segments)} segments")
    print()
    
    # Post-process: merge consecutive segments from same speaker
    merged_segments = []
    current_segment = None
    
    for seg in diarization_segments:
        if current_segment is None:
            current_segment = seg.copy()
        elif (current_segment['speaker'] == seg['speaker'] and 
              seg['start'] - current_segment['end'] < 2.0):  # Merge if gap < 2 seconds
            # Merge with previous
            current_segment['end'] = seg['end']
            current_segment['text'] += " " + seg['text']
        else:
            # Save current and start new
            merged_segments.append(current_segment)
            current_segment = seg.copy()
    
    if current_segment:
        merged_segments.append(current_segment)
    
    print(f"   Merged into {len(merged_segments)} final segments")
    print()
    
    return merged_segments


def run_llm_diarization(
    transcription_file: str = None,
    transcription_text: str = None,
    output_dir: str = None
):
    """
    Run LLM-based diarization on transcription.
    
    Args:
        transcription_file: Path to transcription file (.txt or .json)
        transcription_text: Direct transcription text (if file not provided)
        output_dir: Output directory
    """
    print("=" * 70)
    print("LLM-BASED DIARIZATION")
    print("=" * 70)
    print()
    
    # Load transcription
    if transcription_file:
        print(f"📄 Loading transcription from: {transcription_file}")
        if not os.path.exists(transcription_file):
            print(f"❌ Error: File not found: {transcription_file}")
            sys.exit(1)
        
        with open(transcription_file, 'r', encoding='utf-8') as f:
            if transcription_file.endswith('.json'):
                data = json.load(f)
                transcription_text = data.get('text', '')
            else:
                transcription_text = f.read()
        
        print(f"✅ Loaded {len(transcription_text)} characters")
        print()
    elif transcription_text:
        print(f"📄 Using provided transcription text ({len(transcription_text)} characters)")
        print()
    else:
        print("❌ Error: No transcription provided")
        print("   Use --transcription <file> or provide transcription text")
        sys.exit(1)
    
    # Load FLAN-T5
    print("🤖 Loading FLAN-T5 model...")
    try:
        model_name = Config.LLM_ROLE_IDENTIFICATION_MODEL
        device = Config.LLM_DEVICE
        
        llm_tokenizer = T5Tokenizer.from_pretrained(model_name)
        if llm_tokenizer.pad_token is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token
        
        llm_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        if device == 'cuda' and torch.cuda.is_available():
            llm_model = llm_model.to('cuda')
            print("   Model on CUDA")
        else:
            llm_model = llm_model.to('cpu')
            print("   Model on CPU")
        
        llm_model.eval()
        print(f"✅ FLAN-T5 model loaded: {model_name}")
        print()
    except Exception as e:
        print(f"❌ Error loading FLAN-T5: {e}")
        sys.exit(1)
    
    # Perform diarization
    segments = perform_llm_diarization(transcription_text, llm_model, llm_tokenizer, device)
    
    # Analyze results
    print("=" * 70)
    print("DIARIZATION RESULTS")
    print("=" * 70)
    print()
    
    unique_speakers = set(s['speaker'] for s in segments)
    speaker_stats = {}
    
    for seg in segments:
        speaker = seg['speaker']
        duration = seg['end'] - seg['start']
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {'count': 0, 'total_duration': 0.0}
        
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['total_duration'] += duration
    
    print(f"📊 Found {len(unique_speakers)} speaker(s):")
    print()
    
    for speaker, stats in sorted(speaker_stats.items()):
        print(f"   {speaker}:")
        print(f"      Segments: {stats['count']}")
        print(f"      Total speaking time: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)")
        print()
    
    # Save results
    call_id = f"llm_diarization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{call_id}_diarization.json"
    
    result_data = {
        'call_id': call_id,
        'timestamp': datetime.now().isoformat(),
        'method': 'LLM-based (FLAN-T5)',
        'transcription_source': transcription_file or 'direct_text',
        'segments_count': len(segments),
        'speakers_found': len(unique_speakers),
        'speaker_statistics': {
            speaker: {
                'count': stats['count'],
                'total_duration': stats['total_duration']
            }
            for speaker, stats in speaker_stats.items()
        },
        'segments': segments
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"💾 Results saved to: {output_file}")
    print()
    
    # Save human-readable summary
    summary_file = output_dir / f"{call_id}_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LLM-BASED DIARIZATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Method: FLAN-T5 Text-Based Diarization\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(f"Total Segments: {len(segments)}\n")
        f.write(f"Unique Speakers: {len(unique_speakers)}\n\n")
        
        f.write("Speaker Statistics:\n")
        f.write("-" * 70 + "\n")
        for speaker, stats in sorted(speaker_stats.items()):
            f.write(f"\n{speaker}:\n")
            f.write(f"  Segments: {stats['count']}\n")
            f.write(f"  Total Time: {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} min)\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED SEGMENTS\n")
        f.write("=" * 70 + "\n\n")
        
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}\n")
            f.write(f"   Text: {seg['text'][:200]}{'...' if len(seg['text']) > 200 else ''}\n")
            f.write(f"   Confidence: {seg.get('llm_confidence', 'unknown')}\n")
            f.write("\n")
    
    print(f"📄 Summary saved to: {summary_file}")
    print()
    print("✅ LLM-based diarization completed!")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Perform speaker diarization using LLM (FLAN-T5) on transcription text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use transcription file
  python run_llm_diarization.py transcription_output.txt
  
  # Use JSON transcription
  python run_llm_diarization.py output/call_20251125_160302_transcription.json
        """
    )
    
    parser.add_argument(
        'transcription_file',
        nargs='?',
        default=None,
        help='Path to transcription file (.txt or .json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: output/)'
    )
    
    args = parser.parse_args()
    
    if not args.transcription_file:
        print("❌ Error: Transcription file required")
        print()
        print("Usage:")
        print("  python run_llm_diarization.py <transcription_file>")
        print()
        print("Example:")
        print("  python run_llm_diarization.py transcription_output.txt")
        sys.exit(1)
    
    run_llm_diarization(
        transcription_file=args.transcription_file,
        output_dir=args.output_dir
    )

