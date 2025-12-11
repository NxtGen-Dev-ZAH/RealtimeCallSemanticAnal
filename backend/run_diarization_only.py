#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Diarization Script for Testing
Runs diarization on already transcribed audio without re-running transcription.
This allows you to test different diarization parameters quickly.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from call_analysis.preprocessing import AudioProcessor, TextProcessor
from config import Config

# Load environment variables
load_dotenv()

def run_diarization_only(
    audio_path: str,
    transcription_json: str,
    output_dir: str = None,
    max_speakers: int = 3,
    clustering_threshold: float = 0.3,
    min_segment_duration: float = 1.0,
    speaker_merge_threshold: float = 0.7,
    use_whisperx_builtin: bool = False
):
    """
    Run diarization on already transcribed audio.
    
    Args:
        audio_path: Path to audio file
        transcription_json: Path to existing transcription JSON file
        output_dir: Directory to save results (default: output/)
        max_speakers: Maximum number of speakers to detect
        clustering_threshold: Distance threshold for speaker clustering (0.0-1.0, lower = more clusters)
        min_segment_duration: Minimum segment duration in seconds
        speaker_merge_threshold: Similarity threshold for merging speakers (0.0-1.0)
    """
    print("=" * 70)
    print("DIARIZATION-ONLY TESTING SCRIPT")
    print("=" * 70)
    print()
    
    # Resolve paths
    audio_path = os.path.abspath(audio_path)
    transcription_json = os.path.abspath(transcription_json)
    
    # Set default output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "output")
    
    print(f"📁 Audio file: {audio_path}")
    print(f"📄 Transcription file: {transcription_json}")
    print(f"📂 Output directory: {output_dir}")
    print()
    
    # Verify files exist
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    if not os.path.exists(transcription_json):
        print(f"❌ Error: Transcription file not found: {transcription_json}")
        sys.exit(1)
    
    # Load existing transcription (supports both JSON and TXT)
    print("📖 Loading existing transcription...")
    transcription = None
    transcription_segments = None
    
    if transcription_json.endswith('.json'):
        try:
            with open(transcription_json, 'r', encoding='utf-8') as f:
                transcription = json.load(f)
            transcription_segments = transcription.get('segments', [])
            print(f"✅ Transcription JSON loaded: {len(transcription.get('text', ''))} characters")
            print(f"   Segments with timestamps: {len(transcription_segments)}")
            print("   ✅ JSON format allows text-to-segment mapping (RECOMMENDED)")
        except Exception as e:
            print(f"❌ Error loading transcription JSON: {e}")
            sys.exit(1)
    elif transcription_json.endswith('.txt'):
        try:
            with open(transcription_json, 'r', encoding='utf-8') as f:
                text_content = f.read()
            transcription = {'text': text_content, 'segments': []}
            transcription_segments = []
            print(f"✅ Transcription TXT loaded: {len(text_content)} characters")
            print("   ⚠️  WARNING: TXT format has no timestamps!")
            print("   ⚠️  Text-to-segment mapping will NOT work with TXT files")
            print("   💡 RECOMMENDATION: Use JSON format for better results")
        except Exception as e:
            print(f"❌ Error loading transcription TXT: {e}")
            sys.exit(1)
    else:
        print(f"❌ Error: Unsupported file format. Use .json or .txt")
        sys.exit(1)
    
    # Extract call_id from transcription filename if possible
    transcription_filename = os.path.basename(transcription_json)
    if '_transcription.json' in transcription_filename:
        call_id = transcription_filename.replace('_transcription.json', '')
    else:
        call_id = f"diarization_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🆔 Call ID: {call_id}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # ============================================================
        # STEP 1: Initialize AudioProcessor with custom parameters
        # ============================================================
        print("🔧 Step 1: Initializing AudioProcessor with custom parameters...")
        if max_speakers is not None:
            print(f"   Max speakers (safety limit): {max_speakers}")
        else:
            print(f"   Max speakers: Auto-detect (no limit)")
        print(f"   Clustering threshold: {clustering_threshold} (lower = more speakers, higher = fewer)")
        print(f"   Min segment duration: {min_segment_duration}s")
        print(f"   Speaker merge threshold: {speaker_merge_threshold}")
        print()
        
        if use_whisperx_builtin and not Config.HF_TOKEN:
            print("❌ Error: HF_TOKEN is required for WhisperX 3.x built-in diarization")
            print("   Please set HF_TOKEN in your .env file")
            sys.exit(1)
        
        audio_processor = AudioProcessor(
            model_size=Config.WHISPER_MODEL_SIZE or "base",
            hf_token=Config.HF_TOKEN,
            max_speakers=max_speakers,
            clustering_threshold=clustering_threshold,
            min_segment_duration=min_segment_duration,
            speaker_merge_threshold=speaker_merge_threshold,
            use_whisperx_builtin_diarization=use_whisperx_builtin
        )
        
        if use_whisperx_builtin:
            print("   Using: WhisperX 3.x built-in diarization (Pyannote.audio)")
            print("   ⚠️  This is slower but more accurate than Resemblyzer")
        else:
            print("   Using: WhisperX + Resemblyzer (faster, CPU-friendly)")
        
        print("✅ AudioProcessor initialized")
        print()
        
        # ============================================================
        # STEP 2: Run Diarization
        # ============================================================
        print("🎤 Step 2: Running speaker diarization...")
        print("   (This may take several minutes depending on audio length)")
        print()
        
        diarization_segments = audio_processor.perform_speaker_diarization(audio_path, call_id)
        
        print(f"✅ Diarization completed: {len(diarization_segments)} segments")
        
        # Count unique speakers
        unique_speakers = set(seg.get('speaker', 'Unknown') for seg in diarization_segments)
        print(f"   Unique speakers detected: {len(unique_speakers)}")
        for speaker in sorted(unique_speakers):
            speaker_segments = [s for s in diarization_segments if s.get('speaker') == speaker]
            total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in speaker_segments)
            print(f"   - {speaker}: {len(speaker_segments)} segments, {total_duration:.1f}s total")
        print()
        
        # ============================================================
        # STEP 3: Map Transcription Text to Diarization Segments
        # ============================================================
        print("🔗 Step 3: Mapping transcription text to diarization segments...")
        
        text_processor = TextProcessor()
        segments_with_text = 0
        
        if transcription_segments and len(transcription_segments) > 0:
            # JSON format: Map text using timestamps
            text_processor._assign_text_to_diarization_segments(
                diarization_segments,
                transcription_segments
            )
            segments_with_text = sum(1 for seg in diarization_segments if seg.get('text', '').strip())
            print(f"✅ Text mapping completed: {segments_with_text}/{len(diarization_segments)} segments have text")
        else:
            # TXT format: No timestamps available, cannot map text
            print("⚠️  Cannot map text to segments: TXT format has no timestamps")
            print("   Diarization will work, but segments won't have text content")
            print("   💡 Use JSON format for full functionality")
        
        print()
        
        # ============================================================
        # STEP 4: Save Results
        # ============================================================
        print("💾 Step 4: Saving results...")
        
        # Save diarization JSON
        diarization_output_path = os.path.join(output_dir, f"{call_id}_diarization.json")
        with open(diarization_output_path, 'w', encoding='utf-8') as f:
            json.dump(diarization_segments, f, indent=2, ensure_ascii=False)
        print(f"✅ Diarization saved: {diarization_output_path}")
        
        # Save summary
        summary = {
            'call_id': call_id,
            'audio_path': audio_path,
            'transcription_file': transcription_json,
            'transcription_format': 'json' if transcription_json.endswith('.json') else 'txt',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'max_speakers': max_speakers,
                'clustering_threshold': clustering_threshold,
                'min_segment_duration': min_segment_duration,
                'speaker_merge_threshold': speaker_merge_threshold
            },
            'results': {
                'total_segments': len(diarization_segments),
                'unique_speakers': len(unique_speakers),
                'speakers': {speaker: len([s for s in diarization_segments if s.get('speaker') == speaker])
                            for speaker in unique_speakers},
                'segments_with_text': segments_with_text
            }
        }
        
        summary_path = os.path.join(output_dir, f"{call_id}_diarization_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Summary saved: {summary_path}")
        print()
        
        # ============================================================
        # STEP 5: Display Results
        # ============================================================
        print("=" * 70)
        print("DIARIZATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total segments: {len(diarization_segments)}")
        print(f"Unique speakers: {len(unique_speakers)}")
        print(f"Segments with text: {segments_with_text}")
        print()
        print("Speaker distribution:")
        for speaker in sorted(unique_speakers):
            speaker_segments = [s for s in diarization_segments if s.get('speaker') == speaker]
            total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in speaker_segments)
            avg_duration = total_duration / len(speaker_segments) if speaker_segments else 0
            print(f"  {speaker}:")
            print(f"    - Segments: {len(speaker_segments)}")
            print(f"    - Total duration: {total_duration:.1f}s")
            print(f"    - Avg segment duration: {avg_duration:.1f}s")
        print()
        print("=" * 70)
        print("✅ Diarization testing completed successfully!")
        print("=" * 70)
        print()
        print("💡 TIP: To test different parameters, run this script again with:")
        print(f"   --max-speakers <N>")
        print(f"   --clustering-threshold <0.0-1.0>")
        print(f"   --min-segment-duration <seconds>")
        print(f"   --speaker-merge-threshold <0.0-1.0>")
        print()
        
        return diarization_segments, summary
        
    except Exception as e:
        print(f"❌ Error during diarization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run diarization on already transcribed audio for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with JSON (RECOMMENDED - includes timestamps for text mapping)
  python run_diarization_only.py real_audio.wav output/call_20251125_153129_transcription.json
  
  # With custom parameters
  python run_diarization_only.py real_audio.wav output/call_20251125_153129_transcription.json \\
      --max-speakers 2 \\
      --clustering-threshold 0.4 \\
      --min-segment-duration 0.5
  
  # Using TXT file (NOT RECOMMENDED - no timestamps, text mapping won't work)
  python run_diarization_only.py real_audio.wav transcription_output.txt \\
      --max-speakers 2
  
  # Using WhisperX 3.x built-in diarization (more accurate, slower)
  python run_diarization_only.py real_audio.wav output/call_20251125_160302_transcription.json \\
      --use-whisperx-builtin \\
      --max-speakers 2
        """
    )
    
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to audio file (.wav, .mp3, .m4a)'
    )
    
    parser.add_argument(
        'transcription_file',
        type=str,
        help='Path to existing transcription file (.json recommended, .txt supported but limited)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: output/)'
    )
    
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=None,
        help='Maximum number of speakers as safety limit (None = auto-detect, default: None). '
             'System will automatically determine optimal number based on clustering_threshold.'
    )
    
    parser.add_argument(
        '--clustering-threshold',
        type=float,
        default=0.3,
        help='Distance threshold for speaker clustering, 0.0-1.0, lower = more clusters (default: 0.3)'
    )
    
    parser.add_argument(
        '--min-segment-duration',
        type=float,
        default=1.0,
        help='Minimum segment duration in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--speaker-merge-threshold',
        type=float,
        default=0.7,
        help='Similarity threshold for merging speakers, 0.0-1.0 (default: 0.7)'
    )
    
    parser.add_argument(
        '--use-whisperx-builtin',
        action='store_true',
        help='Use WhisperX 3.x built-in diarization (Pyannote.audio) instead of Resemblyzer. '
             'More accurate but slower. Requires HF_TOKEN.'
    )
    
    args = parser.parse_args()
    
    run_diarization_only(
        audio_path=args.audio_path,
        transcription_json=args.transcription_file,
        output_dir=args.output_dir,
        max_speakers=args.max_speakers,
        clustering_threshold=args.clustering_threshold,
        min_segment_duration=args.min_segment_duration,
        speaker_merge_threshold=args.speaker_merge_threshold,
        use_whisperx_builtin=args.use_whisperx_builtin
    )


if __name__ == '__main__':
    main()

