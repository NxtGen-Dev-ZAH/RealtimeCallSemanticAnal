#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diarization Script with LLM Enhancement
Runs speaker diarization on an audio file with LLM-enhanced role identification and refinement.
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

# Load environment variables
load_dotenv(dotenv_path=backend_dir / ".env")

def find_audio_file(audio_name: str = "real_audio.wav") -> str:
    """
    Find audio file in the project directory.
    
    Args:
        audio_name: Name of the audio file to find
        
    Returns:
        Path to audio file
    """
    # Check current directory
    if os.path.exists(audio_name):
        return os.path.abspath(audio_name)
    
    # Check Call_Analysis directory
    call_analysis_dir = Path(__file__).parent
    audio_path = call_analysis_dir / audio_name
    if audio_path.exists():
        return str(audio_path)
    
    # Check parent directory
    parent_dir = Path(__file__).parent.parent
    audio_path = parent_dir / audio_name
    if audio_path.exists():
        return str(audio_path)
    
    raise FileNotFoundError(f"Audio file '{audio_name}' not found. Please provide the full path.")


def run_diarization_with_llm(
    audio_path: str = None,
    output_dir: str = None,
    use_llm: bool = True,
    use_whisperx_builtin: bool = False,
    clustering_threshold: float = 0.25,
    max_speakers: int = 3,
    speaker_merge_threshold: float = 0.5,
):
    """
    Run diarization with LLM enhancement on an audio file.
    
    Args:
        audio_path: Path to audio file (if None, tries to find real_audio.wav)
        output_dir: Directory to save output (default: output/)
        use_llm: Enable LLM enhancement (default: True)
        use_whisperx_builtin: Use WhisperX built-in diarization (slower but more accurate)
        clustering_threshold: Distance threshold for speaker clustering (0.0-1.0, lower = more speakers, default: 0.25)
        max_speakers: Maximum number of speakers to detect (default: 3)
    """
    print("=" * 70)
    print("DIARIZATION WITH LLM ENHANCEMENT")
    print("=" * 70)
    print()
    
    # Find audio file if not provided
    if audio_path is None:
        print("🔍 Searching for audio file...")
        try:
            audio_path = find_audio_file("real_audio.wav")
            print(f"✅ Found audio file: {audio_path}")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print()
            print("Usage:")
            print(f"  python {sys.argv[0]} <audio_file_path>")
            print()
            print("Or place 'real_audio.wav' in the Call_Analysis directory")
            sys.exit(1)
    else:
        if not os.path.exists(audio_path):
            print(f"❌ Error: Audio file not found: {audio_path}")
            sys.exit(1)
        audio_path = os.path.abspath(audio_path)
    
    # Generate call_id
    call_id = f"diarization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"📁 Audio file: {audio_path}")
    print(f"🆔 Call ID: {call_id}")
    print(f"📂 Output directory: {output_dir}")
    print()
    
    # Check LLM configuration
    if use_llm:
        print("🤖 LLM Enhancement: ENABLED")
        print(f"   Role Identification Model: {Config.LLM_ROLE_IDENTIFICATION_MODEL}")
        print(f"   Refinement Model: {Config.LLM_REFINEMENT_MODEL}")
        print(f"   Device: {Config.LLM_DEVICE}")
        print()
        print("ℹ️  Note: The LLM models (BART-MNLI + FLAN-T5) are PUBLIC models")
        print("   and do NOT require a Hugging Face token. They will download automatically.")
        print()
        print("📋 Processing Pipeline:")
        print("   1. Resemblyzer → Fast speaker embedding extraction")
        print("   2. WhisperX → Accurate word-level timestamps")
        print("   3. BART-MNLI → Role identification (AGENT vs CUSTOMER)")
        print("   4. FLAN-T5 → Correction (fix errors, merge splits)")
        print()
    else:
        print("🤖 LLM Enhancement: DISABLED (using heuristic methods)")
        print()
    
    print(f"⚙️  Diarization Parameters:")
    print(f"   Clustering Threshold: {clustering_threshold} (lower = more speakers detected)")
    print(f"   Max Speakers: {max_speakers}")
    print(f"   Speaker Merge Threshold: {speaker_merge_threshold} (higher = more merging)")
    print()
    
    # Check HF_TOKEN for Pyannote/WhisperX built-in
    if use_whisperx_builtin and not Config.HF_TOKEN:
        print("⚠️  Warning: HF_TOKEN not set. WhisperX built-in diarization requires it.")
        print("   Falling back to WhisperX + Resemblyzer (faster, CPU-friendly)")
        print()
        use_whisperx_builtin = False
    
    try:
        # Initialize AudioProcessor with LLM support
        print("🔧 Initializing AudioProcessor...")
        audio_processor = AudioProcessor(
            model_size=Config.WHISPER_MODEL_SIZE or "base",
            hf_token=Config.HF_TOKEN,
            use_llm_diarization=use_llm,
            llm_role_model=Config.LLM_ROLE_IDENTIFICATION_MODEL,
            llm_refinement_model=Config.LLM_REFINEMENT_MODEL,
            llm_device=Config.LLM_DEVICE,
            use_whisperx_builtin_diarization=use_whisperx_builtin,
            clustering_threshold=clustering_threshold,
            max_speakers=max_speakers,
            speaker_merge_threshold=speaker_merge_threshold,
        )
        print("✅ AudioProcessor initialized")
        print()
        
        # Run diarization
        print("🎤 Running speaker diarization...")
        print("   (This may take several minutes depending on audio length)")
        print()
        print("💡 Progress Indicators:")
        print("   - You'll see 'Processing chunk X/Y' messages")
        print("   - Then 'BART-MNLI' and 'FLAN-T5' messages")
        print("   - Finally '✅ Diarization completed'")
        print()
        print("⏳ Starting now...")
        print()
        
        segments = audio_processor.perform_speaker_diarization(audio_path, call_id)
        
        print()
        print("=" * 70)
        print("DIARIZATION RESULTS")
        print("=" * 70)
        print()
        
        # Analyze results
        unique_speakers = set(seg.get('speaker', 'Unknown') for seg in segments)
        speaker_stats = {}
        
        for seg in segments:
            speaker = seg.get('speaker', 'Unknown')
            duration = seg.get('end', 0) - seg.get('start', 0)
            
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'segments': []
                }
            
            speaker_stats[speaker]['count'] += 1
            speaker_stats[speaker]['total_duration'] += duration
            speaker_stats[speaker]['segments'].append(seg)
        
        print(f"📊 Found {len(unique_speakers)} unique speaker(s):")
        print()
        
        for speaker, stats in sorted(speaker_stats.items()):
            print(f"   {speaker}:")
            print(f"      Segments: {stats['count']}")
            print(f"      Total speaking time: {stats['total_duration']:.1f} seconds ({stats['total_duration']/60:.1f} minutes)")
            print()
        
        # Save detailed results
        output_file = output_dir / f"{call_id}_diarization.json"
        diarization_data = {
            'call_id': call_id,
            'audio_file': audio_path,
            'timestamp': datetime.now().isoformat(),
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
            json.dump(diarization_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Detailed results saved to: {output_file}")
        print()
        
        # Save human-readable summary
        summary_file = output_dir / f"{call_id}_diarization_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SPEAKER DIARIZATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Audio File: {audio_path}\n")
            f.write(f"Call ID: {call_id}\n")
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
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                speaker = seg.get('speaker', 'Unknown')
                text = seg.get('text', '').strip()
                
                f.write(f"{i}. [{start:.1f}s - {end:.1f}s] {speaker}\n")
                if text:
                    f.write(f"   Text: {text}\n")
                f.write("\n")
        
        print(f"📄 Human-readable summary saved to: {summary_file}")
        print()
        print("✅ Diarization completed successfully!")
        print()
        
        return segments
        
    except Exception as e:
        print()
        print("❌ Error during diarization:")
        print(f"   {str(e)}")
        print()
        print("💡 Troubleshooting:")
        print("   1. Ensure audio file is valid (wav, mp3, m4a, flac)")
        print("   2. Check that required models are installed")
        print("   3. For LLM models, ensure internet connection (first download)")
        print("   4. If using WhisperX built-in, ensure HF_TOKEN is set")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run speaker diarization with LLM enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default audio file (real_audio.wav)
  python run_diarization_with_llm.py
  
  # Specify audio file
  python run_diarization_with_llm.py path/to/audio.wav
  
  # Disable LLM enhancement
  python run_diarization_with_llm.py --no-llm
  
  # Use WhisperX built-in diarization (requires HF_TOKEN)
  python run_diarization_with_llm.py --whisperx-builtin
        """
    )
    
    parser.add_argument(
        'audio_file',
        nargs='?',
        default=None,
        help='Path to audio file (default: searches for real_audio.wav)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: output/)'
    )
    
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM enhancement (use heuristic methods only)'
    )
    
    parser.add_argument(
        '--whisperx-builtin',
        action='store_true',
        help='Use WhisperX built-in diarization (slower but more accurate, requires HF_TOKEN)'
    )
    
    parser.add_argument(
        '--clustering-threshold',
        type=float,
        default=0.25,
        help='Clustering threshold for speaker separation (0.0-1.0, lower = more speakers detected, default: 0.25)'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=3,
        help='Maximum number of speakers to detect (default: 3)'
    )
    
    parser.add_argument(
        '--speaker-merge-threshold',
        type=float,
        default=0.5,
        help='Similarity threshold for merging speakers after clustering, 0.0-1.0 (default: 0.5; higher = more merging)'
    )
    
    args = parser.parse_args()
    
    run_diarization_with_llm(
        audio_path=args.audio_file,
        output_dir=args.output_dir,
        use_llm=not args.no_llm,
        use_whisperx_builtin=args.whisperx_builtin,
        clustering_threshold=args.clustering_threshold,
        max_speakers=args.max_speakers,
        speaker_merge_threshold=args.speaker_merge_threshold,
    )

