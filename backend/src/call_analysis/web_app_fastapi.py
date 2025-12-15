"""
FastAPI web application for call analysis system.

This is a FastAPI port of the original Flask-based `web_app.py`, exposing the
same REST API surface so the existing Next.js frontend can keep working:

- GET    /                    -> API root metadata
- GET    /health              -> health check
- GET    /api/conversations   -> demo conversations
- GET    /api/analyze/{id}    -> analyze demo conversation
- GET    /api/analyze-all     -> analyze all demo conversations
- GET    /api/dashboard/{id}  -> generate dashboard HTML
- GET    /api/insights        -> aggregate agent insights
- POST   /api/upload          -> upload & analyze real audio
- POST   /api/analyze         -> start analysis for uploaded call
- GET    /api/results/{id}    -> get analysis results for a call
- GET    /api/status/{id}     -> get analysis status
- GET    /api/history         -> call history
- GET   /api/export/{id}      -> export JSON
- GET   /api/export/{id}/pdf  -> export PDF
- GET   /api/export/{id}/csv  -> export CSV
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    status,
    Depends,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel
from pymongo import MongoClient

from .demo import DemoSystem
from .models import ConversationAnalyzer
from .dashboard import Dashboard

# Import configuration (same pattern as Flask app)
import sys as _sys

_sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import Config  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Call Analysis System API",
        description="FastAPI backend for call analysis, compatible with existing Next.js frontend.",
        version="1.0.0",
    )

    # CORS for frontend (Next.js on localhost:3000)
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize shared components (same as Flask version)
    demo_system = DemoSystem(hf_token=Config.HF_TOKEN)
    analyzer = ConversationAnalyzer()
    dashboard = Dashboard()
    mongo_client = MongoClient(Config.MONGODB_URI)
    db = mongo_client[Config.MONGODB_DATABASE]

    # --------- Helpers ----------

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    # --------- Pydantic models ----------

    class StartAnalysisRequest(BaseModel):
        call_id: str

    # --------- Routes ----------

    @app.get("/")
    def index() -> Dict[str, Any]:
        """API root endpoint."""
        return {
            "message": "Call Analysis System API (FastAPI)",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "upload": "/api/upload",
                "analyze": "/api/analyze",
                "results": "/api/results/{call_id}",
                "status": "/api/status/{call_id}",
                "history": "/api/history",
            },
        }

    @app.get("/api/conversations")
    def get_conversations() -> JSONResponse:
        """Get list of demo conversations."""
        try:
            conversations = [
                {"id": conv["id"], "title": conv["title"], "segments": len(conv["segments"])}
                for conv in demo_system.demo_conversations
            ]
            return JSONResponse(conversations)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Error fetching conversations: {e}")
            raise HTTPException(status_code=500, detail="Failed to load conversations")

    @app.get("/api/analyze/{conversation_id}")
    def analyze_conversation(conversation_id: str) -> JSONResponse:
        """Analyze a specific demo conversation."""
        try:
            result = demo_system.analyze_single_conversation(conversation_id)

            # Mask PII in results (same as Flask)
            for seg in result.get("sentiment_analysis", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            for seg in result.get("segments", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            result["summary"] = dashboard.mask_pii(result.get("summary", ""))

            # Save to MongoDB
            db["analyses"].insert_one(
                {
                    "conversation_id": conversation_id,
                    "result": result,
                    "timestamp": datetime.now(),
                }
            )
            return JSONResponse(result)
        except ValueError as e:
            logger.error(f"Conversation {conversation_id} not found: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Analysis failed for {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail="Analysis failed")

    @app.get("/api/analyze-all")
    def analyze_all() -> JSONResponse:
        """Analyze all demo conversations."""
        try:
            results = demo_system.batch_analyze_all()
            # Mask PII
            for result in results:
                for seg in result.get("sentiment_analysis", []):
                    seg["text"] = dashboard.mask_pii(seg["text"])
                for seg in result.get("segments", []):
                    seg["text"] = dashboard.mask_pii(seg["text"])
                result["summary"] = dashboard.mask_pii(result.get("summary", ""))

            db["batch_analyses"].insert_one(
                {
                    "results": results,
                    "timestamp": datetime.now(),
                }
            )
            return JSONResponse(results)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Batch analysis failed: {e}")
            raise HTTPException(status_code=500, detail="Batch analysis failed")

    @app.get("/api/dashboard/{conversation_id}")
    def get_dashboard(conversation_id: str) -> FileResponse:
        """Generate and serve dashboard HTML for a conversation."""
        try:
            result = demo_system.analyze_single_conversation(conversation_id)
            dashboard_html = demo_system.create_dashboard(result)
            if not os.path.exists(dashboard_html):
                raise HTTPException(status_code=500, detail="Dashboard file not found")
            return FileResponse(
                dashboard_html,
                media_type="text/html",
                filename=os.path.basename(dashboard_html),
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Dashboard generation failed for {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail="Dashboard generation failed")

    @app.get("/api/insights")
    def get_insights() -> JSONResponse:
        """Get agent insights from all conversations."""
        try:
            results = demo_system.batch_analyze_all()
            insights = analyzer.get_agent_insights(results)

            db["insights"].insert_one(
                {
                    "insights": insights,
                    "timestamp": datetime.now(),
                }
            )
            return JSONResponse(insights)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Insights generation failed: {e}")
            raise HTTPException(status_code=500, detail="Insights generation failed")

    @app.post("/api/upload")
    async def upload_audio(file: UploadFile = File(...)) -> JSONResponse:
        """Upload and analyze real audio file."""
        try:
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file selected")
            if not allowed_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type. Use .wav, .mp3, or .m4a",
                )

            # Save file
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            filename = os.path.basename(file.filename)
            audio_path = os.path.join(Config.UPLOAD_FOLDER, filename)

            with open(audio_path, "wb") as f:
                f.write(await file.read())

            # Process audio (full pipeline)
            call_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            transcription = demo_system.audio_processor.transcribe_audio(audio_path, call_id)
            segments = demo_system.audio_processor.perform_speaker_diarization(audio_path, call_id)
            processed_segments = demo_system.text_processor.segment_conversation(
                transcription.get("text", ""),
                segments,
                call_id,
                transcription_segments=transcription.get("segments", []),
            )
            audio_features = demo_system.audio_processor.extract_audio_features(audio_path)

            feature_data = [
                {
                    "audio_features": audio_features,
                    "text_features": {"text": transcription.get("text", "")},
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

            # Mask PII
            for seg in result.get("sentiment_analysis", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            for seg in result.get("segments", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            result["summary"] = dashboard.mask_pii(result.get("summary", ""))

            # Save to MongoDB - calls collection for frontend
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

            # Also save to analyses collection for backward compatibility
            db.analyses.insert_one(
                {
                    "conversation_id": call_id,
                    "result": result,
                    "audio_path": audio_path,
                    "timestamp": datetime.now(),
                }
            )

            return JSONResponse(
                {
                    "status": "success",
                    "message": "Audio analyzed successfully",
                    "call_id": call_id,
                    "filename": filename,
                    "size": os.path.getsize(audio_path),
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Audio upload failed: {e}")
            raise HTTPException(status_code=500, detail="Audio analysis failed")

    @app.get("/api/export/{conversation_id}")
    def export_results(conversation_id: str) -> FileResponse:
        """Export analysis results as JSON file."""
        try:
            result = demo_system.analyze_single_conversation(conversation_id)

            # Mask PII
            for seg in result.get("sentiment_analysis", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            for seg in result.get("segments", []):
                seg["text"] = dashboard.mask_pii(seg["text"])
            result["summary"] = dashboard.mask_pii(result.get("summary", ""))

            export_data = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat(),
                "analysis": result,
            }

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            export_path = os.path.join(output_dir, f"{conversation_id}_export.json")

            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            db["exports"].insert_one(
                {
                    "conversation_id": conversation_id,
                    "export_path": export_path,
                    "data": export_data,
                    "timestamp": datetime.now(),
                }
            )

            return FileResponse(
                export_path,
                media_type="application/json",
                filename=os.path.basename(export_path),
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Export failed for {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail="Export failed")

    @app.get("/api/export/{call_id}/pdf")
    def export_pdf(call_id: str) -> FileResponse:
        """Export analysis results as PDF report."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
        except ImportError:  # pragma: no cover - optional dependency
            logger.error("ReportLab not installed. Install with: uv pip install reportlab")
            raise HTTPException(status_code=500, detail="PDF export requires reportlab package")

        try:
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            pdf_path = os.path.join(export_dir, f"{call_id}_report.pdf")

            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story: List[Any] = []

            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=18,
                spaceAfter=30,
                alignment=1,
            )
            story.append(Paragraph("Call Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # Call info
            story.append(Paragraph("Call Information", styles["Heading2"]))
            call_info = [
                ["Call ID:", call_id],
                ["Filename:", call_doc.get("filename", "N/A")],
                [
                    "Date:",
                    call_doc.get("timestamp", "N/A").strftime("%Y-%m-%d %H:%M:%S")
                    if call_doc.get("timestamp")
                    else "N/A",
                ],
                ["Duration:", f"{call_doc.get('duration', 0)} seconds"],
                ["Participants:", str(call_doc.get("participants", 2))],
            ]

            call_table = Table(call_info, colWidths=[2 * inch, 4 * inch])
            call_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("BACKGROUND", (1, 0), (1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(call_table)
            story.append(Spacer(1, 20))

            # Summary
            story.append(Paragraph("Analysis Results", styles["Heading2"]))
            story.append(Paragraph("Summary", styles["Heading3"]))
            summary_data = [
                ["Average Sentiment:", f"{call_doc.get('avg_sentiment', 0):.2f}"],
                ["Sale Probability:", f"{call_doc.get('sale_probability', 0) * 100:.1f}%"],
                [
                    "Overall Sentiment:",
                    "Positive"
                    if call_doc.get("avg_sentiment", 0) > 0.6
                    else "Neutral"
                    if call_doc.get("avg_sentiment", 0) > 0.4
                    else "Negative",
                ],
            ]

            summary_table = Table(summary_data, colWidths=[2 * inch, 4 * inch])
            summary_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.lightblue),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(summary_table)
            story.append(Spacer(1, 20))

            # Emotion distribution
            story.append(Paragraph("Emotion Distribution", styles["Heading3"]))
            emotions = call_doc.get("emotions", {})
            if emotions:
                emotion_data = [["Emotion", "Percentage"]]
                for emotion, value in emotions.items():
                    emotion_data.append([emotion.title(), f"{value * 100:.1f}%"])

                emotion_table = Table(emotion_data, colWidths=[2 * inch, 2 * inch])
                emotion_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 12),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )
                story.append(emotion_table)

            story.append(Spacer(1, 20))

            # Key phrases
            story.append(Paragraph("Key Phrases", styles["Heading3"]))
            key_phrases = call_doc.get("key_phrases", {"positive": [], "negative": []})

            if key_phrases.get("positive"):
                story.append(Paragraph("Positive Phrases:", styles["Heading4"]))
                for phrase in key_phrases["positive"][:5]:
                    story.append(Paragraph(f"• {phrase}", styles["Normal"]))

            if key_phrases.get("negative"):
                story.append(Paragraph("Negative Phrases:", styles["Heading4"]))
                for phrase in key_phrases["negative"][:5]:
                    story.append(Paragraph(f"• {phrase}", styles["Normal"]))

            doc.build(story)

            return FileResponse(
                pdf_path,
                media_type="application/pdf",
                filename=os.path.basename(pdf_path),
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"PDF export failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="PDF export failed")

    @app.get("/api/export/{call_id}/csv")
    def export_csv(call_id: str) -> Response:
        """Export analysis results as CSV data."""
        try:
            import csv
            import io
        except ImportError:  # pragma: no cover
            raise HTTPException(status_code=500, detail="CSV export requires csv and io modules")

        try:
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            output = io.StringIO()
            writer = csv.writer(output)

            writer.writerow(["Call Analysis Data"])
            writer.writerow(["Call ID", call_id])
            writer.writerow(["Filename", call_doc.get("filename", "N/A")])
            writer.writerow(
                [
                    "Date",
                    call_doc.get("timestamp", "N/A").strftime("%Y-%m-%d %H:%M:%S")
                    if call_doc.get("timestamp")
                    else "N/A",
                ]
            )
            writer.writerow(["Duration (seconds)", call_doc.get("duration", 0)])
            writer.writerow(["Participants", call_doc.get("participants", 2)])
            writer.writerow([])

            writer.writerow(["SUMMARY"])
            writer.writerow(["Average Sentiment", f"{call_doc.get('avg_sentiment', 0):.2f}"])
            writer.writerow(["Sale Probability (%)", f"{call_doc.get('sale_probability', 0) * 100:.1f}"])
            writer.writerow(
                [
                    "Overall Sentiment",
                    "Positive"
                    if call_doc.get("avg_sentiment", 0) > 0.6
                    else "Neutral"
                    if call_doc.get("avg_sentiment", 0) > 0.4
                    else "Negative",
                ]
            )
            writer.writerow([])

            writer.writerow(["EMOTION DISTRIBUTION"])
            writer.writerow(["Emotion", "Percentage"])
            emotions = call_doc.get("emotions", {})
            for emotion, value in emotions.items():
                writer.writerow([emotion.title(), f"{value * 100:.1f}%"])
            writer.writerow([])

            writer.writerow(["SENTIMENT SCORES OVER TIME"])
            writer.writerow(["Timestamp", "Sentiment Score"])
            for i, score_data in enumerate(call_doc.get("sentiment_scores", [])):
                timestamp = score_data.get("timestamp", i)
                score = score_data.get("score", 0)
                writer.writerow([f"T{timestamp}", f"{score:.2f}"])
            writer.writerow([])

            writer.writerow(["KEY PHRASES"])
            key_phrases = call_doc.get("key_phrases", {"positive": [], "negative": []})

            writer.writerow(["Positive Phrases"])
            for phrase in key_phrases.get("positive", []):
                writer.writerow([phrase])

            writer.writerow(["Negative Phrases"])
            for phrase in key_phrases.get("negative", []):
                writer.writerow([phrase])

            csv_content = output.getvalue()
            output.close()

            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{call_id}_analysis.csv"'},
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"CSV export failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="CSV export failed")

    @app.post("/api/analyze")
    def start_analysis(payload: StartAnalysisRequest) -> JSONResponse:
        """Start analysis for uploaded file (status update only)."""
        try:
            call_id = payload.call_id

            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            db.calls.update_one(
                {"call_id": call_id},
                {"$set": {"status": "processing", "updated_at": datetime.now()}},
            )

            return JSONResponse(
                {
                    "message": "Analysis started",
                    "call_id": call_id,
                    "status": "processing",
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Analysis start failed: {e}")
            raise HTTPException(status_code=500, detail="Analysis start failed")

    @app.get("/api/results/{call_id}")
    def get_results(call_id: str) -> JSONResponse:
        """Get analysis results for a call."""
        try:
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            return JSONResponse(
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
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Results retrieval failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="Results retrieval failed")

    @app.get("/api/status/{call_id}")
    def get_status(call_id: str) -> JSONResponse:
        """Get analysis status for a call."""
        try:
            call_doc = db.calls.find_one({"call_id": call_id})
            if not call_doc:
                raise HTTPException(status_code=404, detail="Call not found")

            return JSONResponse(
                {
                    "call_id": call_id,
                    "status": call_doc.get("status", "unknown"),
                    "progress": call_doc.get("progress", 0),
                }
            )
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover
            logger.error(f"Status check failed for {call_id}: {e}")
            raise HTTPException(status_code=500, detail="Status check failed")

    @app.get("/api/history")
    def get_history() -> JSONResponse:
        """Get call history."""
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

            return JSONResponse(calls)
        except Exception as e:  # pragma: no cover
            logger.error(f"History retrieval failed: {e}")
            raise HTTPException(status_code=500, detail="History retrieval failed")

    @app.get("/health")
    def health_check() -> JSONResponse:
        """Health check endpoint."""
        try:
            db.command("ping")
            return JSONResponse(
                {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                }
            )
        except Exception as e:  # pragma: no cover
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                {"status": "unhealthy", "error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    return app


app = create_app()


