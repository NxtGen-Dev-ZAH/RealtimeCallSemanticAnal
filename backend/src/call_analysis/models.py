"""
ML models for sentiment analysis, emotion detection, and sale prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Text sentiment analysis using BERT/DistilBERT"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.is_trained = False
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained BERT model"""
        try:
            logger.info(f"Loading {self.model_name} model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load BERT model: {e}")
            logger.info("Using simplified sentiment analysis for demo")
    
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract BERT embeddings from texts
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if self.tokenizer and self.model:
            try:
                embeddings = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt", 
                                          truncation=True, padding=True, max_length=512)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Use [CLS] token embedding
                        embedding = outputs.last_hidden_state[:, 0, :].numpy()
                        embeddings.append(embedding.flatten())
                return np.array(embeddings)
            except Exception as e:
                logger.error(f"BERT embedding extraction failed: {e}")
                return self._demo_embeddings(texts)
        else:
            return self._demo_embeddings(texts)
    
    def _demo_embeddings(self, texts: List[str]) -> np.ndarray:
        """Demo embeddings for prototype"""
        # Generate random embeddings with consistent dimensions
        return np.random.randn(len(texts), 768)
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Simple rule-based sentiment for demo
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'love', 'like', 'interested', 'yes', 'sure', 'definitely', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
                         'no', 'not', 'never', 'worst', 'disappointed', 'frustrated']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score
        total_words = len(text_lower.split())
        if total_words == 0:
            sentiment_score = 0
        else:
            sentiment_score = (pos_count - neg_count) / total_words
        
        # Classify sentiment
        if sentiment_score > 0.1:
            sentiment = "positive"
            confidence = min(abs(sentiment_score), 1.0)
        elif sentiment_score < -0.1:
            sentiment = "negative"
            confidence = min(abs(sentiment_score), 1.0)
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(sentiment_score)
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "confidence": confidence,
            "positive_words": pos_count,
            "negative_words": neg_count
        }
    
    def analyze_conversation_sentiment(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for each conversation segment
        
        Args:
            segments: List of conversation segments
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        for segment in segments:
            text = segment.get('text', '')
            sentiment_result = self.analyze_sentiment(text)
            
            result = {
                "start_time": segment.get('start_time', 0),
                "end_time": segment.get('end_time', 0),
                "speaker": segment.get('speaker', 'Unknown'),
                "text": text,
                **sentiment_result
            }
            results.append(result)
        
        return results


class EmotionDetector:
    """Audio emotion detection using CNN+LSTM"""
    
    def __init__(self):
        """Initialize emotion detector"""
        self.model = None
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
        self.is_trained = False
    
    def _create_model(self, input_shape: Tuple[int, ...]) -> nn.Module:
        """
        Create CNN+LSTM model for emotion detection
        
        Args:
            input_shape: Input feature shape
            
        Returns:
            PyTorch model
        """
        class EmotionCNN(nn.Module):
            def __init__(self, input_dim, num_classes=7):
                super(EmotionCNN, self).__init__()
                self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
                
                self.pool = nn.MaxPool1d(2)
                self.dropout = nn.Dropout(0.5)
                
                self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(256, num_classes)
                self.softmax = nn.Softmax(dim=1)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = self.dropout(x)
                
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = self.dropout(x)
                
                x = torch.relu(self.conv3(x))
                x = self.pool(x)
                x = self.dropout(x)
                
                # Reshape for LSTM
                x = x.transpose(1, 2)
                lstm_out, _ = self.lstm(x)
                
                # Global average pooling
                x = torch.mean(lstm_out, dim=1)
                x = self.fc(x)
                x = self.softmax(x)
                
                return x
        
        return EmotionCNN(input_shape[0], len(self.emotion_labels))
    
    def detect_emotion(self, audio_features: Dict) -> Dict:
        """
        Detect emotion from audio features
        
        Args:
            audio_features: Dictionary of audio features
            
        Returns:
            Dictionary with emotion detection results
        """
        # For demo purposes, simulate emotion detection
        return self._demo_emotion_detection(audio_features)
    
    def _demo_emotion_detection(self, audio_features: Dict) -> Dict:
        """Demo emotion detection"""
        # Simulate emotion probabilities
        emotions = self.emotion_labels
        probabilities = np.random.dirichlet(np.ones(len(emotions)))
        
        # Find dominant emotion
        dominant_emotion_idx = np.argmax(probabilities)
        dominant_emotion = emotions[dominant_emotion_idx]
        confidence = probabilities[dominant_emotion_idx]
        
        return {
            "emotion": dominant_emotion,
            "confidence": confidence,
            "probabilities": dict(zip(emotions, probabilities))
        }
    
    def detect_conversation_emotions(self, segments: List[Dict], audio_features: Dict) -> List[Dict]:
        """
        Detect emotions for conversation segments
        
        Args:
            segments: List of conversation segments
            audio_features: Audio features for the entire conversation
            
        Returns:
            List of emotion detection results
        """
        results = []
        base_emotion = self.detect_emotion(audio_features)
        
        for i, segment in enumerate(segments):
            # Vary emotion slightly for each segment
            emotion_variation = np.random.normal(0, 0.1, len(self.emotion_labels))
            probabilities = np.array(list(base_emotion['probabilities'].values())) + emotion_variation
            probabilities = np.maximum(probabilities, 0)  # Ensure non-negative
            probabilities = probabilities / np.sum(probabilities)  # Normalize
            
            dominant_emotion_idx = np.argmax(probabilities)
            dominant_emotion = self.emotion_labels[dominant_emotion_idx]
            confidence = probabilities[dominant_emotion_idx]
            
            result = {
                "start_time": segment.get('start_time', 0),
                "end_time": segment.get('end_time', 0),
                "speaker": segment.get('speaker', 'Unknown'),
                "emotion": dominant_emotion,
                "confidence": confidence,
                "probabilities": dict(zip(self.emotion_labels, probabilities))
            }
            results.append(result)
        
        return results


class SalePredictor:
    """Sale probability prediction using XGBoost/LSTM"""
    
    def __init__(self):
        """Initialize sale predictor"""
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        self.model_type = "xgboost"  # or "lstm"
    
    def train(self, X: np.ndarray, y: np.ndarray, model_type: str = "xgboost"):
        """
        Train the sale prediction model
        
        Args:
            X: Feature matrix
            y: Target labels (0 or 1 for sale/no sale)
            model_type: Type of model to use ("xgboost" or "lstm")
        """
        self.model_type = model_type
        
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X, y)
            self.feature_importance = self.model.feature_importances_
        
        elif model_type == "lstm":
            # For demo, use a simple neural network
            self.model = self._create_lstm_model(X.shape[1])
            # In a real implementation, you would train the LSTM here
            logger.info("LSTM model created (demo mode - not trained)")
        
        self.is_trained = True
        logger.info(f"Sale prediction model trained using {model_type}")
    
    def _create_lstm_model(self, input_dim: int) -> nn.Module:
        """Create LSTM model for sale prediction"""
        class SaleLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_layers=2):
                super(SaleLSTM, self).__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                  batch_first=True, dropout=0.3)
                self.fc = nn.Linear(hidden_dim, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Take the last output
                last_output = lstm_out[:, -1, :]
                output = self.fc(last_output)
                return self.sigmoid(output)
        
        return SaleLSTM(input_dim)
    
    def predict_sale_probability(self, features: np.ndarray) -> Dict:
        """
        Predict sale probability
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            # Demo prediction
            return self._demo_prediction(features)
        
        if self.model_type == "xgboost":
            probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
        else:
            # For LSTM demo
            probability = np.random.uniform(0.2, 0.8)
        
        return {
            "sale_probability": probability,
            "prediction": "sale" if probability > 0.5 else "no_sale",
            "confidence": abs(probability - 0.5) * 2,  # Distance from 0.5
            "feature_importance": self.feature_importance.tolist() if self.feature_importance is not None else None
        }
    
    def _demo_prediction(self, features: np.ndarray) -> Dict:
        """Demo sale prediction"""
        # Simulate prediction based on feature characteristics
        # Higher values in certain features indicate higher sale probability
        demo_probability = np.random.uniform(0.1, 0.9)
        
        return {
            "sale_probability": demo_probability,
            "prediction": "sale" if demo_probability > 0.5 else "no_sale",
            "confidence": abs(demo_probability - 0.5) * 2,
            "feature_importance": np.random.rand(features.shape[0]).tolist() if len(features) > 0 else None
        }
    
    def get_feature_importance(self) -> Dict:
        """
        Get feature importance for the trained model
        
        Returns:
            Dictionary with feature importance
        """
        if self.feature_importance is not None:
            return {
                "importance": self.feature_importance.tolist(),
                "top_features": np.argsort(self.feature_importance)[-10:].tolist()  # Top 10 features
            }
        else:
            return {"importance": [], "top_features": []}


class ConversationAnalyzer:
    """Combines all models for comprehensive conversation analysis"""
    
    def __init__(self):
        """Initialize conversation analyzer"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.sale_predictor = SalePredictor()
        
        # Train demo models
        self._train_demo_models()
    
    def _train_demo_models(self):
        """Train demo models with synthetic data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # Train sale predictor
        self.sale_predictor.train(X, y, model_type="xgboost")
        logger.info("Demo models trained with synthetic data")
    
    def analyze_conversation(self, audio_path: str = None, text_data: str = None, 
                           segments: List[Dict] = None) -> Dict:
        """
        Perform comprehensive conversation analysis
        
        Args:
            audio_path: Path to audio file (optional)
            text_data: Raw text data (optional)
            segments: Pre-processed conversation segments (optional)
            
        Returns:
            Comprehensive analysis results
        """
        # For demo purposes, use provided segments or create demo data
        if segments is None:
            segments = self._create_demo_segments()
        
        # Analyze sentiment for each segment
        sentiment_results = self.sentiment_analyzer.analyze_conversation_sentiment(segments)
        
        # Analyze emotions (demo audio features)
        demo_audio_features = {
            "mfcc": np.random.randn(13, 100),
            "spectral_centroid": np.random.randn(1, 100),
            "duration": 50.0
        }
        emotion_results = self.emotion_detector.detect_conversation_emotions(segments, demo_audio_features)
        
        # Prepare features for sale prediction
        demo_features = np.random.randn(50)  # 50-dimensional feature vector
        sale_prediction = self.sale_predictor.predict_sale_probability(demo_features)
        
        # Calculate conversation-level metrics
        conversation_metrics = self._calculate_conversation_metrics(sentiment_results, emotion_results)
        
        return {
            "conversation_id": f"demo_{np.random.randint(1000, 9999)}",
            "duration": max([s.get('end_time', 0) for s in segments]),
            "segments": len(segments),
            "sentiment_analysis": sentiment_results,
            "emotion_analysis": emotion_results,
            "sale_prediction": sale_prediction,
            "conversation_metrics": conversation_metrics,
            "summary": self._generate_summary(sentiment_results, emotion_results, sale_prediction)
        }

    def _create_demo_segments(self) -> List[Dict]:
        """Create demo conversation segments"""
        demo_segments = [
            {
                "start_time": 0, "end_time": 10, "speaker": "Customer",
                "text": "Hello, I'm interested in your insurance policy. Can you tell me more about the coverage?"
            },
            {
                "start_time": 10, "end_time": 20, "speaker": "Agent", 
                "text": "Of course! I'd be happy to help. Our comprehensive policy covers home, auto, and life insurance with great rates."
            },
            {
                "start_time": 20, "end_time": 30, "speaker": "Customer",
                "text": "That sounds interesting. What are the premium costs and what's included in the basic plan?"
            },
            {
                "start_time": 30, "end_time": 40, "speaker": "Agent",
                "text": "The basic plan starts at $120 per month and includes liability coverage, emergency assistance, and 24/7 customer support."
            },
            {
                "start_time": 40, "end_time": 50, "speaker": "Customer",
                "text": "That's reasonable. I think I'd like to proceed with this policy. How do we get started?"
            }
        ]
        return demo_segments
    
    def _calculate_conversation_metrics(self, sentiment_results: List[Dict], 
                                      emotion_results: List[Dict]) -> Dict:
        """Calculate conversation-level metrics"""
        if not sentiment_results or not emotion_results:
            return {}
        
        # Sentiment metrics
        sentiment_scores = [r['score'] for r in sentiment_results]
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_trend = np.polyfit(range(len(sentiment_scores)), sentiment_scores, 1)[0]
        
        # Emotion metrics
        emotions = [r['emotion'] for r in emotion_results]
        emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Speaker analysis
        customer_segments = [r for r in sentiment_results if r['speaker'] == 'Customer']
        agent_segments = [r for r in sentiment_results if r['speaker'] == 'Agent']
        
        customer_avg_sentiment = np.mean([r['score'] for r in customer_segments]) if customer_segments else 0
        agent_avg_sentiment = np.mean([r['score'] for r in agent_segments]) if agent_segments else 0
        
        return {
            "average_sentiment": avg_sentiment,
            "sentiment_trend": sentiment_trend,
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "customer_sentiment": customer_avg_sentiment,
            "agent_sentiment": agent_avg_sentiment,
            "sentiment_volatility": np.std(sentiment_scores),
            "conversation_flow_score": abs(sentiment_trend) * len(sentiment_results)
        }
    
    
    def _generate_summary(self, sentiment_results: List[Dict], emotion_results: List[Dict], 
                         sale_prediction: Dict) -> str:
        """Generate a natural language summary of the conversation"""
        sale_prob = sale_prediction.get('sale_probability', 0)
        avg_sentiment = np.mean([r['score'] for r in sentiment_results]) if sentiment_results else 0
        
        if sale_prob > 0.7:
            sale_outlook = "very positive"
        elif sale_prob > 0.5:
            sale_outlook = "positive"
        elif sale_prob > 0.3:
            sale_outlook = "moderate"
        else:
            sale_outlook = "low"
        
        if avg_sentiment > 0.1:
            mood = "positive"
        elif avg_sentiment < -0.1:
            mood = "negative"
        else:
            mood = "neutral"
        
        summary = f"""
        Conversation Analysis Summary:
        - Overall sentiment: {mood} (score: {avg_sentiment:.2f})
        - Sale probability: {sale_prob:.1%} ({sale_outlook} outlook)
        - Dominant emotion: {emotion_results[0]['emotion'] if emotion_results else 'neutral'}
        - Conversation segments: {len(sentiment_results)}
        - Key insights: Customer engagement appears {'high' if sale_prob > 0.6 else 'moderate' if sale_prob > 0.4 else 'low'}
        """
        
        return summary.strip()
    
    def batch_analyze(self, conversations: List[Dict]) -> List[Dict]:
        """
        Analyze multiple conversations in batch
        
        Args:
            conversations: List of conversation data
            
        Returns:
            List of analysis results
        """
        results = []
        for conv in conversations:
            result = self.analyze_conversation(segments=conv.get('segments'))
            results.append(result)
        
        return results
    
    def get_agent_insights(self, conversation_results: List[Dict]) -> Dict:
        """
        Generate insights for call center agents
        
        Args:
            conversation_results: List of conversation analysis results
            
        Returns:
            Dictionary with agent insights and recommendations
        """
        if not conversation_results:
            return {}
        
        # Aggregate metrics across conversations
        sale_probabilities = [r['sale_prediction']['sale_probability'] for r in conversation_results]
        avg_sale_prob = np.mean(sale_probabilities)
        
        sentiment_scores = []
        for result in conversation_results:
            sentiment_scores.extend([s['score'] for s in result['sentiment_analysis']])
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        # Generate recommendations
        recommendations = []
        if avg_sale_prob < 0.4:
            recommendations.append("Focus on building rapport and addressing customer concerns")
        if avg_sentiment < -0.2:
            recommendations.append("Use more positive language and focus on benefits")
        if avg_sentiment > 0.2:
            recommendations.append("Maintain a neutral tone and avoid over-promising")
        
        return {
            "average_sale_probability": avg_sale_prob,
            "average_sentiment": avg_sentiment,
            "recommendations": recommendations
        }