"""
Original Flask web application for call analysis system (backup).

This file was created automatically when migrating the API to FastAPI
so that the previous Flask implementation is preserved for reference.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from pymongo import MongoClient

from .demo import DemoSystem
from .models import ConversationAnalyzer
from .dashboard import Dashboard

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})
Config.init_app(app)

demo_system = DemoSystem(hf_token=Config.HF_TOKEN)
analyzer = ConversationAnalyzer()
dashboard = Dashboard()
mongo_client = MongoClient(Config.MONGODB_URI)
db = mongo_client[Config.MONGODB_DATABASE]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return jsonify({"message": "Call Analysis System API (Flask backup)"}), 200


@app.route("/api/upload", methods=["POST"])
def upload_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use .wav, .mp3, or .m4a"}), 400

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(audio_path)

        call_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        transcription = demo_system.audio_processor.transcribe_audio(audio_path, call_id)
        segments = demo_system.audio_processor.perform_speaker_diarization(audio_path, call_id)
        processed_segments = demo_system.text_processor.segment_conversation(
            transcription["text"],
            segments,
            call_id,
            transcription_segments=transcription.get("segments", []),
        )
        audio_features = demo_system.audio_processor.extract_audio_features(audio_path)

        feature_data = [
            {
                "audio_features": audio_features,
                "text_features": {"text": transcription["text"]},
                "segments": processed_segments,
            }
        ]
        fused_features = demo_system.feature_extractor.fit_transform(feature_data)[0]

        result = demo_system.analyzer.analyze_conversation(
            audio_path=audio_path,
            segments=processed_segments,
            features=fused_features,
            call_id=call_id,
        )

        for seg in result.get("sentiment_analysis", []):
            seg["text"] = dashboard.mask_pii(seg["text"])
        for seg in result.get("segments", []):
            seg["text"] = dashboard.mask_pii(seg["text"])
        result["summary"] = dashboard.mask_pii(result.get("summary", ""))

        call_doc = {
            "call_id": call_id,
            "filename": filename,
            "audio_path": audio_path,
            "timestamp": datetime.now(),
            "status": "completed",
            "progress": 100,
            "duration": result.get("duration", 0),
            "participants": result.get("participants", 2),
            "avg_sentiment": result.get("avg_sentiment", 0),
            "sale_probability": result.get("sale_probability", 0),
            "sentiment_scores": result.get("sentiment_scores", []),
            "emotions": result.get("emotions", {}),
            "key_phrases": result.get("key_phrases", {"positive": [], "negative": []}),
            "result": result,
        }

        db.calls.insert_one(call_doc)

        return jsonify(
            {
                "status": "success",
                "message": "Audio analyzed successfully",
                "call_id": call_id,
                "filename": filename,
                "size": os.path.getsize(audio_path),
            }
        )
    except Exception as e:
        logger.error(f"Audio upload failed: {e}")
        return jsonify({"error": "Audio analysis failed"}), 500


@app.route("/api/results/<call_id>")
def get_results(call_id):
    try:
        call_doc = db.calls.find_one({"call_id": call_id})
        if not call_doc:
            return jsonify({"error": "Call not found"}), 404

        return jsonify(
            {
                "call_id": call_id,
                "sentiment_scores": call_doc.get("sentiment_scores", []),
                "emotions": call_doc.get("emotions", {}),
                "sale_probability": call_doc.get("sale_probability", 0),
                "key_phrases": call_doc.get("key_phrases", {"positive": [], "negative": []}),
                "summary": {
                    "avg_sentiment": call_doc.get("avg_sentiment", 0),
                    "total_duration": call_doc.get("duration", 0),
                    "participants": call_doc.get("participants", 2),
                },
            }
        )
    except Exception as e:
        logger.error(f"Results retrieval failed for {call_id}: {e}")
        return jsonify({"error": "Results retrieval failed"}), 500


@app.route("/api/status/<call_id>")
def get_status(call_id):
    try:
        call_doc = db.calls.find_one({"call_id": call_id})
        if not call_doc:
            return jsonify({"error": "Call not found"}), 404

        return jsonify(
            {
                "call_id": call_id,
                "status": call_doc.get("status", "unknown"),
                "progress": call_doc.get("progress", 0),
            }
        )
    except Exception as e:
        logger.error(f"Status check failed for {call_id}: {e}")
        return jsonify({"error": "Status check failed"}), 500


@app.route("/api/history")
def get_history():
    try:
        calls = list(
            db.calls.find(
                {},
                {
                    "call_id": 1,
                    "filename": 1,
                    "timestamp": 1,
                    "duration": 1,
                    "avg_sentiment": 1,
                    "sale_probability": 1,
                    "participants": 1,
                    "status": 1,
                },
            ).sort("timestamp", -1)
        )

        for call in calls:
            call["_id"] = str(call["_id"])

        return jsonify(calls)
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        return jsonify({"error": "History retrieval failed"}), 500


@app.route("/health")
def health_check():
    try:
        db.command("ping")
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


