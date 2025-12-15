#!/usr/bin/env python3
"""
Transcribe audio file using Whisper
Simple script to extract text from audio file
"""

import sys
import os
from pathlib import Path

# Add src to path (now we're in backend/scripts/, so go up one level to backend/, then to src)
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, src_dir)

from dotenv import load_dotenv
from call_analysis.preprocessing import AudioProcessor

# Load environment variables
load_dotenv()

def transcribe_audio_file(audio_path: str, model_size: str = "base"):
    """
    Transcribe audio file to text using Whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
    """
    print("=" * 60)
    print("WHISPER AUDIO TRANSCRIPTION")
    print("=" * 60)
    print()
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file not found: {audio_path}")
        return
    
    print(f"📁 Audio file: {audio_path}")
    file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
    print(f"📊 File size: {file_size:.2f} MB")
    print()
    
    # Initialize AudioProcessor
    print(f"🔧 Initializing Whisper (model: {model_size})...")
    print("   (This may take a moment on first run)")
    print()
    
    try:
        # Get HF token from environment (optional for Whisper, required for Pyannote)
        hf_token = os.getenv('HF_TOKEN', None)
        processor = AudioProcessor(model_size=model_size, hf_token=hf_token)
        print("✅ Whisper initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize Whisper: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   1. Make sure 'openai-whisper' is installed: pip install openai-whisper")
        print("   2. Check if you have enough disk space for model download")
        print("   3. Try smaller model: model_size='tiny' or 'base'")
        return
    
    # Transcribe audio
    print("🎤 Starting transcription...")
    print("   (This may take several minutes depending on audio length)")
    print()
    
    try:
        call_id = "transcription_001"
        transcription = processor.transcribe_audio(audio_path, call_id)
        
        # Display results
        text = transcription.get('text', '')
        language = transcription.get('language', 'unknown')
        duration = transcription.get('duration', 0)
        segments = transcription.get('segments', [])
        
        print("✅ Transcription completed!")
        print()
        print("=" * 60)
        print("TRANSCRIPTION RESULTS")
        print("=" * 60)
        print(f"📝 Language: {language}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"📊 Number of segments: {len(segments)}")
        print(f"📝 Text length: {len(text)} characters")
        print()
        
        # Display full transcript
        print("📄 Full Transcript:")
        print("-" * 60)
        if text:
            print(text)
        else:
            print("(No speech detected in audio file)")
        print("-" * 60)
        print()
        
        # Show segments with timestamps
        if segments:
            print("📋 Segments with Timestamps:")
            print("-" * 60)
            for i, seg in enumerate(segments[:10], 1):  # Show first 10 segments
                start = seg.get('start', 0)
                end = seg.get('end', 0)
                seg_text = seg.get('text', '').strip()
                if seg_text:
                    print(f"{i}. [{start:.1f}s - {end:.1f}s]: {seg_text}")
            if len(segments) > 10:
                print(f"... and {len(segments) - 10} more segments")
            print("-" * 60)
            print()
        
        # Save to file - use output directory relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(script_dir)
        project_root = os.path.dirname(backend_dir)
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "transcription_output.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("WHISPER TRANSCRIPTION OUTPUT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Audio file: {audio_path}\n")
                f.write(f"Language: {language}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Number of segments: {len(segments)}\n")
                f.write(f"Text length: {len(text)} characters\n\n")
                f.write("=" * 60 + "\n")
                f.write("FULL TRANSCRIPT\n")
                f.write("=" * 60 + "\n\n")
                f.write(text)
                f.write("\n\n")
                f.write("=" * 60 + "\n")
                f.write("SEGMENTS WITH TIMESTAMPS\n")
                f.write("=" * 60 + "\n\n")
                for i, seg in enumerate(segments, 1):
                    start = seg.get('start', 0)
                    end = seg.get('end', 0)
                    seg_text = seg.get('text', '').strip()
                    f.write(f"{i}. [{start:.1f}s - {end:.1f}s]: {seg_text}\n")
            
            print(f"💾 Transcript saved to: {output_file}")
            print()
        except Exception as e:
            print(f"⚠️  Could not save transcript to file: {e}")
            print()
        
        print("✅ Done!")
        print()
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        print()
        print("💡 Common issues:")
        print("   1. Audio file format not supported (use WAV, MP3, or M4A)")
        print("   2. Audio file is corrupted or empty")
        print("   3. Insufficient memory (try smaller model: 'tiny' or 'base')")
        print("   4. Audio quality too poor for Whisper to detect speech")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    # Get project root (go up from backend/scripts/ to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(backend_dir)
    
    # Default to real_audio.wav in project root
    audio_file = os.path.join(project_root, "real_audio.wav")
    
    # Check if file exists
    if not os.path.exists(audio_file):
        # Try just the filename in project root
        filename = "real_audio.wav"
        audio_file = os.path.join(project_root, filename)
        if not os.path.exists(audio_file):
            print("❌ Error: real_audio.wav not found in project root")
            print()
            print("Usage:")
            print("  python transcribe_audio.py")
            print("  OR")
            print("  python transcribe_audio.py path/to/audio.wav")
            print()
            sys.exit(1)
    
    # Check for command line argument
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        # If relative path, make it absolute or relative to project root
        if not os.path.isabs(audio_file):
            # Try relative to project root first
            test_path = os.path.join(project_root, audio_file)
            if os.path.exists(test_path):
                audio_file = test_path
            # Otherwise use as-is (relative to current working directory)
    
    # Optional: specify model size as second argument
    model_size = "base"
    if len(sys.argv) > 2:
        model_size = sys.argv[2]
    
    transcribe_audio_file(audio_file, model_size)

