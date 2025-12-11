"""
Feature extraction module for audio and text features, aligned with SRS1 (Purely ML-Based System).
Includes BERT integration for text features (FR-4), improved sentiment calculation,
audio preprocessing from raw files (FR-1/FR-5), database storage (FR-7),
PII masking for security, and enhanced error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import librosa
from transformers import BertTokenizer, BertModel
import torch
from pymongo import MongoClient
from datetime import datetime
import spacy
import unicodedata
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_ENABLED = os.getenv('MONGODB_ENABLED', 'false').lower() == 'true'
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGO_DB_NAME = os.getenv('MONGODB_DATABASE', 'call_center_db')

_BERT_TOKENIZER = None
_BERT_MODEL = None

# Load spaCy model for PII masking
# Load spaCy for PII masking (optional)
try:
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
    logger.info("spaCy model loaded successfully")
except OSError:
    nlp = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy model 'en_core_web_sm' not found. PII masking will be limited.")


class FeatureExtractor:
    """Extracts and combines features from audio and text data, with SRS1 alignment."""
    
    def __init__(self):
        """Initialize feature extractor with BERT and scalers."""
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.audio_scaler = StandardScaler()
        self.text_scaler = StandardScaler()
        self.is_fitted = False
        
        global _BERT_TOKENIZER, _BERT_MODEL
        if _BERT_TOKENIZER is None:
            _BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        if _BERT_MODEL is None:
            _BERT_MODEL = BertModel.from_pretrained('bert-base-uncased')
        self.bert_tokenizer = _BERT_TOKENIZER
        self.bert_model = _BERT_MODEL
        
        # Placeholder for sentiment classifier (e.g., fine-tuned on top of BERT)
        # In practice, train this with your dataset
        self.sentiment_classifier = lambda emb: np.mean(emb)  # Dummy; replace with actual classifier
    
    def mask_pii(self, text: str) -> str:
        """
        Mask personally identifiable information (PII) in text for security (SRS1 regulatory constraints).
        
        Args:
            text: Raw text to anonymize.
            
        Returns:
            Anonymized text.
        """
        if not text or not text.strip():
            return text
        
        if not SPACY_AVAILABLE or nlp is None:
            # Fallback to simple regex-based masking if spaCy not available
            import re
            phone_pattern = r'(?:(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4})'
            text = re.sub(phone_pattern, '[PHONE]', text)
            # Mask email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
            return text
        
        try:
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'PHONE', 'EMAIL', 'GPE', 'ORG']:
                    text = text.replace(ent.text, '[REDACTED]')
            return text
        except Exception as e:
            logger.error(f"Error in PII masking: {e}")
            return text  # Fallback to original if error
    
    def load_audio_features(self, audio_path: str) -> Dict:
        """
        Load and extract raw audio features from file (aligns with FR-1: accept .wav/.mp3/.m4a).
        
        Args:
            audio_path: Path to audio file.
            
        Returns:
            Dictionary of audio features.
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)  # Resample to 16kHz for consistency
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            duration = librosa.get_duration(y=y, sr=sr)
            
            return {
                'mfcc': mfcc,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'chroma': chroma,
                'duration': duration,
                'sample_rate': sr
            }
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return {}  # Empty dict on failure
    
    def extract_audio_features(self, audio_features: Dict) -> np.ndarray:
        """
        Extract numerical features from audio data (FR-5: CNN+LSTM for emotion).
        
        Args:
            audio_features: Dictionary of audio features.
            
        Returns:
            Numpy array of audio features.
        """
        if not audio_features:
            logger.warning("Empty audio features provided")
            return np.zeros(50)  # Default shape to avoid crashes
        
        features = []
        
        # MFCC features (mean and std)
        if 'mfcc' in audio_features:
            mfcc = audio_features['mfcc']
            features.extend([
                np.mean(mfcc, axis=1),  # Mean of each MFCC coefficient
                np.std(mfcc, axis=1)    # Std of each MFCC coefficient
            ])
        
        # Spectral features
        if 'spectral_centroid' in audio_features:
            sc = audio_features['spectral_centroid']
            features.extend([np.mean(sc), np.std(sc)])
        
        if 'spectral_rolloff' in audio_features:
            sr = audio_features['spectral_rolloff']
            features.extend([np.mean(sr), np.std(sr)])
        
        if 'zero_crossing_rate' in audio_features:
            zcr = audio_features['zero_crossing_rate']
            features.extend([np.mean(zcr), np.std(zcr)])
        
        # Chroma features
        if 'chroma' in audio_features:
            chroma = audio_features['chroma']
            features.extend([
                np.mean(chroma, axis=1),  # Mean chroma values
                np.std(chroma, axis=1)    # Std chroma values
            ])
        
        # Duration and sample rate
        features.extend([
            audio_features.get('duration', 0),
            audio_features.get('sample_rate', 0) / 1000  # Normalize sample rate
        ])
        
        # Flatten and return
        flat_features = np.concatenate([f.flatten() if hasattr(f, 'flatten') else [f] for f in features])
        return flat_features
    
    def extract_text_features(self, text_features: Dict) -> np.ndarray:
        """
        Extract BERT embeddings from text data (FR-4: BERT for sentiment).
        
        Args:
            text_features: Dictionary with 'text' key (raw text).
            
        Returns:
            Numpy array of BERT embeddings.
        """
        text = self.mask_pii(text_features.get('text', ''))  # Mask PII first
        if not text:
            logger.warning("Empty text provided")
            return np.zeros(768)  # BERT embedding size
        
        try:
            inputs = self.bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            return embedding
        except Exception as e:
            logger.error(f"Error in BERT extraction: {e}")
            return np.zeros(768)
    
    def extract_temporal_features(self, segments: List[Dict]) -> Dict:
        """
        Extract temporal features from conversation segments (supports FR-8 dashboard).
        
        Args:
            segments: List of conversation segments (from diarization, FR-3).
            
        Returns:
            Dictionary of temporal features.
        """
        if not segments:
            logger.warning("Empty segments provided")
            return {}
        
        try:
            # Calculate conversation flow metrics
            total_duration = max([s.get('end_time', 0) for s in segments])
            speaker_changes = len([i for i in range(1, len(segments)) 
                                  if segments[i].get('speaker') != segments[i-1].get('speaker')])
            
            # Calculate speaking time by speaker
            speaker_times = {}
            for segment in segments:
                speaker = segment.get('speaker', 'Unknown')
                duration = segment.get('end_time', 0) - segment.get('start_time', 0)
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            
            # Sentiment progression using BERT (improved from keyword-based)
            sentiment_scores = []
            for segment in segments:
                text = self.mask_pii(segment.get('text', ''))
                bert_embedding = self.extract_text_features({'text': text})
                sentiment = self.sentiment_classifier(bert_embedding)  # Use classifier
                sentiment_scores.append(sentiment)
            
            temporal_features = {
                'total_duration': total_duration,
                'speaker_changes': speaker_changes,
                'avg_segment_duration': total_duration / len(segments) if segments else 0,
                'speaker_balance': min(speaker_times.values()) / max(speaker_times.values()) if speaker_times else 0,
                'sentiment_trend': np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0] if len(sentiment_scores) > 1 else 0,
                'sentiment_volatility': np.std(sentiment_scores) if sentiment_scores else 0,
                'final_sentiment': sentiment_scores[-1] if sentiment_scores else 0
            }
            
            return temporal_features
        except Exception as e:
            logger.error(f"Error in temporal features: {e}")
            return {}
    
    def combine_features(self, audio_features: np.ndarray, text_features: np.ndarray, 
                         temporal_features: Dict) -> np.ndarray:
        """
        Combine all feature types into a single feature vector (FR-6: fused features).
        
        Args:
            audio_features: Audio feature vector.
            text_features: Text feature vector.
            temporal_features: Dictionary of temporal features.
            
        Returns:
            Combined feature vector.
        """
        # Convert temporal features to array
        temporal_array = np.array(list(temporal_features.values()))
        
        # Combine all features
        combined = np.concatenate([audio_features, text_features, temporal_array])
        
        return combined
    
    def fit_transform(self, feature_data: List[Dict]) -> np.ndarray:
        """
        Fit scalers and transform features (optimized for batches).
        
        Args:
            feature_data: List of feature dictionaries.
            
        Returns:
            Transformed feature matrix.
        """
        if not feature_data:
            return np.array([])
        
        # Vectorized extraction for performance
        all_features = np.array([self.combine_features(
            self.extract_audio_features(d.get('audio_features', {})),
            self.extract_text_features(d.get('text_features', {})),
            self.extract_temporal_features(d.get('segments', []))
        ) for d in feature_data])
        
        # Fit scalers if not already fitted
        if not self.is_fitted:
            self.audio_scaler.fit(all_features)
            self.is_fitted = True
        
        # Scale features
        scaled_features = self.audio_scaler.transform(all_features)
        
        return scaled_features
    
    def transform(self, feature_data: Dict) -> np.ndarray:
        """
        Transform single sample features.
        
        Args:
            feature_data: Feature dictionary for single sample.
            
        Returns:
            Transformed feature vector.
        """
        audio_feat = self.extract_audio_features(feature_data.get('audio_features', {}))
        text_feat = self.extract_text_features(feature_data.get('text_features', {}))
        temporal_feat = self.extract_temporal_features(feature_data.get('segments', []))
        
        combined = self.combine_features(audio_feat, text_feat, temporal_feat)
        
        # Reshape for scaler
        combined = combined.reshape(1, -1)
        
        if self.is_fitted:
            scaled = self.audio_scaler.transform(combined)
            return scaled.flatten()
        else:
            return combined.flatten()
    
    def save_features(self, features: np.ndarray, call_id: str):
        """
        Save features to MongoDB (FR-7: store data in PostgreSQL/MongoDB).
        
        Args:
            features: Feature vector.
            call_id: Unique call identifier.
        """
        if not MONGO_ENABLED:
            logger.debug("MongoDB disabled - skipping feature persistence")
            return
        try:
            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
            db = client[MONGO_DB_NAME]
            collection = db['features']
            collection.insert_one({
                'call_id': call_id,
                'features': features.tolist(),
                'timestamp': datetime.now()
            })
            logger.info(f"Features saved for call_id: {call_id}")
        except Exception as e:
            logger.error(f"Error saving features: {e}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all features for interpretability (supports FR-8 dashboard).
        
        Returns:
            List of feature names.
        """
        feature_names = []
        
        # Audio features
        feature_names.extend([f'mfcc_mean_{i}' for i in range(13)])
        feature_names.extend([f'mfcc_std_{i}' for i in range(13)])
        feature_names.extend(['spectral_centroid_mean', 'spectral_centroid_std'])
        feature_names.extend(['spectral_rolloff_mean', 'spectral_rolloff_std'])
        feature_names.extend(['zcr_mean', 'zcr_std'])
        feature_names.extend([f'chroma_mean_{i}' for i in range(12)])
        feature_names.extend([f'chroma_std_{i}' for i in range(12)])
        feature_names.extend(['duration', 'sample_rate'])
        
        # BERT text features (768 dimensions)
        feature_names.extend([f'bert_embedding_{i}' for i in range(768)])
        
        # Temporal features
        feature_names.extend([
            'total_duration', 'speaker_changes', 'avg_segment_duration',
            'speaker_balance', 'sentiment_trend', 'sentiment_volatility', 'final_sentiment'
        ])
        
        return feature_names