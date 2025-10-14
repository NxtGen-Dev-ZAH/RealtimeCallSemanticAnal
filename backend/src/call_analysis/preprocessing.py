"""
Data preprocessing module for audio and text processing, aligned with SRS1 (Purely ML-Based System).
Handles audio loading (FR-1), Whisper transcription (FR-2), Pyannote diarization (FR-3),
audio feature extraction (FR-5), text preprocessing for BERT (FR-4), and MongoDB storage (FR-7).
"""

import os
import librosa
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import whisper
from pyannote.audio import Pipeline
import soundfile as sf
from pydub import AudioSegment
import logging
from jiwer import wer
import spacy
from pymongo import MongoClient
from datetime import datetime
from transformers import BertTokenizer
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy for PII masking (optional)
try:
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
    logger.info("spaCy model loaded successfully")
except OSError:
    nlp = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy model 'en_core_web_sm' not found. PII masking will be limited.")


class AudioProcessor:
    """Handles audio preprocessing including transcription and speaker diarization."""
    
    def __init__(self, model_size: str = "base", hf_token: str = None):
        """
        Initialize audio processor.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
            hf_token: Hugging Face token for Pyannote.audio.
        """
        self.model_size = model_size
        self.hf_token = hf_token
        self.whisper_model = None
        self.diarization_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and Pyannote models (FR-2, FR-3)."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
            
            logger.info("Loading Pyannote.audio diarization pipeline")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=self.hf_token
            )
            logger.info("Audio models loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def validate_audio_format(self, audio_path: str) -> str:
        """
        Validate and convert audio format (FR-1: .wav, .mp3, .m4a).
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Path to validated/converted audio.
        """
        try:
            if not audio_path.endswith(('.wav', '.mp3', '.m4a')):
                logger.info(f"Converting {audio_path} to .wav")
                audio = AudioSegment.from_file(audio_path)
                new_path = audio_path.rsplit('.', 1)[0] + '.wav'
                audio.export(new_path, format='wav')
                return new_path
            return audio_path
        except Exception as e:
            logger.error(f"Audio format validation failed: {e}")
            raise ValueError(f"Unsupported audio format: {audio_path}")
    
    def transcribe_audio(self, audio_path: str, call_id: str) -> Dict:
        """
        Transcribe audio file to text (FR-2: Whisper, WER ≤15%).
        
        Args:
            audio_path: Path to audio file.
            call_id: Unique call identifier for storage.
            
        Returns:
            Dictionary with transcription results.
        """
        try:
            audio_path = self.validate_audio_format(audio_path)
            result = self.whisper_model.transcribe(audio_path)
            transcription = {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", "en")
            }
            self.save_transcription(transcription, call_id)  # FR-7
            return transcription
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def validate_transcription(self, transcription: str, reference: str) -> float:
        """
        Validate transcription quality (FR-2: WER ≤15%).
        
        Args:
            transcription: Transcribed text.
            reference: Ground-truth text.
            
        Returns:
            Word Error Rate (WER).
        """
        try:
            return wer(reference, transcription)
        except Exception as e:
            logger.error(f"WER calculation failed: {e}")
            return 1.0  # Assume worst-case
    
    def perform_speaker_diarization(self, audio_path: str, call_id: str) -> List[Dict]:
        """
        Perform speaker diarization to identify speakers (FR-3: Pyannote.audio).
        
        Args:
            audio_path: Path to audio file.
            call_id: Unique call identifier for storage.
            
        Returns:
            List of speaker segments with timestamps.
        """
        try:
            audio_path = self.validate_audio_format(audio_path)
            diarization = self.diarization_pipeline(audio_path)
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "text": ""  # To be filled by TextProcessor
                })
            self.save_diarization(segments, call_id)  # FR-7
            return segments
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
    
    def extract_audio_features(self, audio_path: str) -> Dict:
        """
        Extract audio features for emotion detection (FR-5: CNN+LSTM).
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Dictionary of audio features.
        """
        try:
            audio_path = self.validate_audio_format(audio_path)
            y, sr = librosa.load(audio_path, sr=16000)  # Match Whisper's default
            features = {
                "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
                "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
                "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
                "zero_crossing_rate": librosa.feature.zero_crossing_rate(y),
                "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
                "mel_spectrogram": librosa.feature.melspectrogram(y=y, sr=sr),
                "duration": librosa.get_duration(y=y, sr=sr),
                "sample_rate": sr
            }
            return features
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            raise
    
    def save_transcription(self, transcription: Dict, call_id: str):
        """
        Save transcription to MongoDB (FR-7).
        
        Args:
            transcription: Transcription dictionary.
            call_id: Unique call identifier.
        """
        try:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['call_center_db']
            collection = db['transcriptions']
            collection.insert_one({
                'call_id': call_id,
                'transcription': transcription,
                'timestamp': datetime.now()
            })
            logger.info(f"Transcription saved for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Error saving transcription: {e}")
    
    def save_diarization(self, segments: List[Dict], call_id: str):
        """
        Save diarization segments to MongoDB (FR-7).
        
        Args:
            segments: List of diarization segments.
            call_id: Unique call identifier.
        """
        try:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['call_center_db']
            collection = db['diarization']
            collection.insert_one({
                'call_id': call_id,
                'segments': segments,
                'timestamp': datetime.now()
            })
            logger.info(f"Diarization saved for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Error saving diarization: {e}")
    
    def process_batch(self, audio_paths: List[str], call_ids: List[str]) -> List[Dict]:
        """
        Process multiple audio files in parallel (performance optimization).
        
        Args:
            audio_paths: List of audio file paths.
            call_ids: List of unique call identifiers.
            
        Returns:
            List of transcription dictionaries.
        """
        try:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.transcribe_audio, audio_paths, call_ids))
            return results
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise


class TextProcessor:
    """Handles text preprocessing and cleaning."""
    
    def __init__(self):
        """Initialize text processor with BERT tokenizer and stop words."""
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        ])
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def mask_pii(self, text: str) -> str:
        """
        Mask PII in text for security (SRS1 regulatory constraints).
        
        Args:
            text: Raw text input.
            
        Returns:
            Anonymized text.
        """
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'PHONE', 'EMAIL', 'GPE', 'ORG']:
                    text = text.replace(ent.text, '[REDACTED]')
            return text
        except Exception as e:
            logger.error(f"Error in PII masking: {e}")
            return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text input.
            
        Returns:
            Cleaned text.
        """
        import re
        text = self.mask_pii(text)  # Mask PII first
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_text(self, text: str) -> Dict:
        """
        Tokenize text for BERT (FR-4).
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary with BERT tokens.
        """
        text = self.mask_pii(text)
        tokens = self.bert_tokenizer(
            text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        )
        return {'tokens': tokens, 'raw_text': text}
    
    def extract_text_features(self, text: str) -> Dict:
        """
        Extract text features for sentiment analysis (FR-4: BERT-compatible).
        
        Args:
            text: Input text.
            
        Returns:
            Dictionary with BERT tokens and basic features.
        """
        text = self.mask_pii(text)
        tokens = self.tokenize_text(text)
        raw_text = tokens['raw_text']
        word_tokens = raw_text.split()
        
        features = {
            'tokens': tokens['tokens'],  # For BERT
            'word_count': len(word_tokens),
            'char_count': len(raw_text),
            'avg_word_length': np.mean([len(word) for word in word_tokens]) if word_tokens else 0,
            'sentence_count': len([s for s in raw_text.split('.') if s.strip()]),
            'exclamation_count': raw_text.count('!'),
            'question_count': raw_text.count('?'),
            'capital_ratio': sum(1 for c in raw_text if c.isupper()) / len(raw_text) if raw_text else 0
        }
        return features
    
    def segment_conversation(self, text: str, segments: List[Dict], call_id: str) -> List[Dict]:
        """
        Segment conversation by speaker and time (supports FR-8).
        
        Args:
            text: Full conversation text.
            segments: List of diarization segments.
            call_id: Unique call identifier.
            
        Returns:
            List of processed segments.
        """
        processed_segments = []
        text = self.mask_pii(text)
        
        for segment in segments:
            segment_text = self.mask_pii(segment.get('text', ''))
            processed_segment = {
                'start_time': segment.get('start', 0),
                'end_time': segment.get('end', 0),
                'speaker': segment.get('speaker', 'Unknown'),
                'text': segment_text,
                'features': self.extract_text_features(segment_text)
            }
            processed_segments.append(processed_segment)
        
        self.save_segments(processed_segments, call_id)  # FR-7
        return processed_segments
    
    def save_segments(self, segments: List[Dict], call_id: str):
        """
        Save conversation segments to MongoDB (FR-7).
        
        Args:
            segments: List of processed segments.
            call_id: Unique call identifier.
        """
        try:
            client = MongoClient('mongodb://localhost:27017/')
            db = client['call_center_db']
            collection = db['segments']
            collection.insert_one({
                'call_id': call_id,
                'segments': segments,
                'timestamp': datetime.now()
            })
            logger.info(f"Segments saved for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Error saving segments: {e}")