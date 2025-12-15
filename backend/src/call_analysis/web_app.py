"""
Flask web application for call analysis system, aligned with SRS1.
Provides API endpoints and UI for conversation analysis, dashboard, and insights.
Integrates full pipeline, stores results in MongoDB (FR-7), and masks PII.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
from typing import Dict, List
from werkzeug.utils import secure_filename
from pymongo import MongoClient

from .demo import DemoSystem
from .models import ConversationAnalyzer
from .dashboard import Dashboard

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for frontend connections
CORS(app, resources={
    r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]},
    r"/health": {"origins": "*"}
})
Config.init_app(app)

# Initialize components
demo_system = DemoSystem(hf_token=Config.HF_TOKEN)
analyzer = ConversationAnalyzer()
dashboard = Dashboard()
mongo_client = MongoClient(Config.MONGODB_URI)
db = mongo_client[Config.MONGODB_DATABASE]

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """API root endpoint."""
    return jsonify({
        'message': 'Call Analysis System API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'upload': '/api/upload',
            'analyze': '/api/analyze',
            'results': '/api/results/<call_id>',
            'status': '/api/status/<call_id>',
            'history': '/api/history'
        }
    })

@app.route('/api/conversations')
def get_conversations():
    """Get list of demo conversations."""
    try:
        conversations = [
            {'id': conv['id'], 'title': conv['title'], 'segments': len(conv['segments'])}
            for conv in demo_system.demo_conversations
        ]
        return jsonify(conversations)
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return jsonify({'error': 'Failed to load conversations'}), 500

@app.route('/api/analyze/<conversation_id>')
def analyze_conversation(conversation_id: str):
    """Analyze a specific conversation."""
    try:
        result = demo_system.analyze_single_conversation(conversation_id)
        # Mask PII in results
        for seg in result.get('sentiment_analysis', []):
            seg['text'] = dashboard.mask_pii(seg['text'])
        for seg in result.get('segments', []):
            seg['text'] = dashboard.mask_pii(seg['text'])
        result['summary'] = dashboard.mask_pii(result.get('summary', ''))
        
        # Save to MongoDB (FR-7)
        db['analyses'].insert_one({
            'conversation_id': conversation_id,
            'result': result,
            'timestamp': datetime.now()
        })
        return jsonify(result)
    except ValueError as e:
        logger.error(f"Conversation {conversation_id} not found: {e}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Analysis failed for {conversation_id}: {e}")
        return jsonify({'error': 'Analysis failed'}), 500

@app.route('/api/analyze-all')
def analyze_all():
    """Analyze all demo conversations."""
    try:
        results = demo_system.batch_analyze_all()
        # Mask PII
        for result in results:
            for seg in result.get('sentiment_analysis', []):
                seg['text'] = dashboard.mask_pii(seg['text'])
            for seg in result.get('segments', []):
                seg['text'] = dashboard.mask_pii(seg['text'])
            result['summary'] = dashboard.mask_pii(result.get('summary', ''))
        
        # Save to MongoDB (FR-7)
        db['batch_analyses'].insert_one({
            'results': results,
            'timestamp': datetime.now()
        })
        return jsonify(results)
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return jsonify({'error': 'Batch analysis failed'}), 500

@app.route('/api/dashboard/<conversation_id>')
def get_dashboard(conversation_id: str):
    """Generate and serve dashboard for a conversation."""
    try:
        result = demo_system.analyze_single_conversation(conversation_id)
        dashboard_html = demo_system.create_dashboard(result)
        return send_file(dashboard_html)
    except Exception as e:
        logger.error(f"Dashboard generation failed for {conversation_id}: {e}")
        return jsonify({'error': 'Dashboard generation failed'}), 500

@app.route('/api/insights')
def get_insights():
    """Get agent insights from all conversations."""
    try:
        results = demo_system.batch_analyze_all()
        insights = analyzer.get_agent_insights(results)
        # Save to MongoDB (FR-7)
        db['insights'].insert_one({
            'insights': insights,
            'timestamp': datetime.now()
        })
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Insights generation failed: {e}")
        return jsonify({'error': 'Insights generation failed'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Upload and analyze real audio file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use .wav, .mp3, or .m4a'}), 400
        
        # Save file
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)
        
        # Process audio (full pipeline)
        call_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        transcription = demo_system.audio_processor.transcribe_audio(audio_path, call_id)
        segments = demo_system.audio_processor.perform_speaker_diarization(audio_path, call_id)
        processed_segments = demo_system.text_processor.segment_conversation(
            transcription['text'],
            segments,
            call_id,
            transcription_segments=transcription.get('segments', [])
        )
        audio_features = demo_system.audio_processor.extract_audio_features(audio_path)
        
        feature_data = [{'audio_features': audio_features, 'text_features': {'text': transcription['text']}, 'segments': processed_segments}]
        fused_features = demo_system.feature_extractor.fit_transform(feature_data)[0]
        
        # Analyze
        result = demo_system.analyzer.analyze_conversation(
            audio_path=audio_path,
            segments=processed_segments,
            features=fused_features,
            call_id=call_id
        )
        
        # Mask PII
        for seg in result.get('sentiment_analysis', []):
            seg['text'] = dashboard.mask_pii(seg['text'])
        for seg in result.get('segments', []):
            seg['text'] = dashboard.mask_pii(seg['text'])
        result['summary'] = dashboard.mask_pii(result.get('summary', ''))
        
        # Save to MongoDB (FR-7) - calls collection for frontend
        call_doc = {
            'call_id': call_id,
            'filename': filename,
            'audio_path': audio_path,
            'timestamp': datetime.now(),
            'status': 'completed',
            'progress': 100,
            'duration': result.get('duration', 0),
            'participants': result.get('participants', 2),
            'avg_sentiment': result.get('avg_sentiment', 0),
            'sale_probability': result.get('sale_probability', 0),
            'sentiment_scores': result.get('sentiment_scores', []),
            'emotions': result.get('emotions', {}),
            'key_phrases': result.get('key_phrases', {'positive': [], 'negative': []}),
            'result': result
        }
        
        db.calls.insert_one(call_doc)
        
        # Also save to analyses collection for backward compatibility
        db.analyses.insert_one({
            'conversation_id': call_id,
            'result': result,
            'audio_path': audio_path,
            'timestamp': datetime.now()
        })
        
        return jsonify({
            'status': 'success',
            'message': 'Audio analyzed successfully',
            'call_id': call_id,
            'filename': filename,
            'size': os.path.getsize(audio_path)
        })
    except Exception as e:
        logger.error(f"Audio upload failed: {e}")
        return jsonify({'error': 'Audio analysis failed'}), 500

@app.route('/api/export/<conversation_id>')
def export_results(conversation_id: str):
    """Export analysis results as JSON."""
    try:
        result = demo_system.analyze_single_conversation(conversation_id)
        # Mask PII
        for seg in result.get('sentiment_analysis', []):
            seg['text'] = dashboard.mask_pii(seg['text'])
        for seg in result.get('segments', []):
            seg['text'] = dashboard.mask_pii(seg['text'])
        result['summary'] = dashboard.mask_pii(result.get('summary', ''))
        
        export_data = {
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': result
        }
        
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, f'{conversation_id}_export.json')
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Save to MongoDB (FR-7)
        db['exports'].insert_one({
            'conversation_id': conversation_id,
            'export_path': export_path,
            'data': export_data,
            'timestamp': datetime.now()
        })
        
        return send_file(export_path, as_attachment=True)
    except Exception as e:
        logger.error(f"Export failed for {conversation_id}: {e}")
        return jsonify({'error': 'Export failed'}), 500

@app.route('/api/export/<call_id>/pdf')
def export_pdf(call_id: str):
    """Export analysis results as PDF report."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        # Get results from database
        call_doc = db.calls.find_one({'call_id': call_id})
        if not call_doc:
            return jsonify({'error': 'Call not found'}), 404
        
        # Create PDF
        export_dir = 'exports'
        os.makedirs(export_dir, exist_ok=True)
        pdf_path = os.path.join(export_dir, f'{call_id}_report.pdf')
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Call Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Call Information
        story.append(Paragraph("Call Information", styles['Heading2']))
        call_info = [
            ['Call ID:', call_id],
            ['Filename:', call_doc.get('filename', 'N/A')],
            ['Date:', call_doc.get('timestamp', 'N/A').strftime('%Y-%m-%d %H:%M:%S') if call_doc.get('timestamp') else 'N/A'],
            ['Duration:', f"{call_doc.get('duration', 0)} seconds"],
            ['Participants:', str(call_doc.get('participants', 2))]
        ]
        
        call_table = Table(call_info, colWidths=[2*inch, 4*inch])
        call_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(call_table)
        story.append(Spacer(1, 20))
        
        # Analysis Results
        story.append(Paragraph("Analysis Results", styles['Heading2']))
        
        # Summary
        story.append(Paragraph("Summary", styles['Heading3']))
        summary_data = [
            ['Average Sentiment:', f"{call_doc.get('avg_sentiment', 0):.2f}"],
            ['Sale Probability:', f"{call_doc.get('sale_probability', 0) * 100:.1f}%"],
            ['Overall Sentiment:', 'Positive' if call_doc.get('avg_sentiment', 0) > 0.6 else 'Neutral' if call_doc.get('avg_sentiment', 0) > 0.4 else 'Negative']
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Emotion Distribution
        story.append(Paragraph("Emotion Distribution", styles['Heading3']))
        emotions = call_doc.get('emotions', {})
        if emotions:
            emotion_data = [['Emotion', 'Percentage']]
            for emotion, value in emotions.items():
                emotion_data.append([emotion.title(), f"{value * 100:.1f}%"])
            
            emotion_table = Table(emotion_data, colWidths=[2*inch, 2*inch])
            emotion_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(emotion_table)
        
        story.append(Spacer(1, 20))
        
        # Key Phrases
        story.append(Paragraph("Key Phrases", styles['Heading3']))
        key_phrases = call_doc.get('key_phrases', {'positive': [], 'negative': []})
        
        if key_phrases.get('positive'):
            story.append(Paragraph("Positive Phrases:", styles['Heading4']))
            for phrase in key_phrases['positive'][:5]:  # Limit to 5 phrases
                story.append(Paragraph(f"• {phrase}", styles['Normal']))
        
        if key_phrases.get('negative'):
            story.append(Paragraph("Negative Phrases:", styles['Heading4']))
            for phrase in key_phrases['negative'][:5]:  # Limit to 5 phrases
                story.append(Paragraph(f"• {phrase}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        return send_file(pdf_path, as_attachment=True, mimetype='application/pdf')
        
    except ImportError:
        logger.error("ReportLab not installed. Install with: pip install reportlab")
        return jsonify({'error': 'PDF export requires reportlab package'}), 500
    except Exception as e:
        logger.error(f"PDF export failed for {call_id}: {e}")
        return jsonify({'error': 'PDF export failed'}), 500

@app.route('/api/export/<call_id>/csv')
def export_csv(call_id: str):
    """Export analysis results as CSV data."""
    try:
        import csv
        import io
        
        # Get results from database
        call_doc = db.calls.find_one({'call_id': call_id})
        if not call_doc:
            return jsonify({'error': 'Call not found'}), 404
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Call Analysis Data'])
        writer.writerow(['Call ID', call_id])
        writer.writerow(['Filename', call_doc.get('filename', 'N/A')])
        writer.writerow(['Date', call_doc.get('timestamp', 'N/A').strftime('%Y-%m-%d %H:%M:%S') if call_doc.get('timestamp') else 'N/A'])
        writer.writerow(['Duration (seconds)', call_doc.get('duration', 0)])
        writer.writerow(['Participants', call_doc.get('participants', 2)])
        writer.writerow([])  # Empty row
        
        # Summary data
        writer.writerow(['SUMMARY'])
        writer.writerow(['Average Sentiment', f"{call_doc.get('avg_sentiment', 0):.2f}"])
        writer.writerow(['Sale Probability (%)', f"{call_doc.get('sale_probability', 0) * 100:.1f}"])
        writer.writerow(['Overall Sentiment', 'Positive' if call_doc.get('avg_sentiment', 0) > 0.6 else 'Neutral' if call_doc.get('avg_sentiment', 0) > 0.4 else 'Negative'])
        writer.writerow([])  # Empty row
        
        # Emotion distribution
        writer.writerow(['EMOTION DISTRIBUTION'])
        writer.writerow(['Emotion', 'Percentage'])
        emotions = call_doc.get('emotions', {})
        for emotion, value in emotions.items():
            writer.writerow([emotion.title(), f"{value * 100:.1f}%"])
        writer.writerow([])  # Empty row
        
        # Sentiment scores over time
        writer.writerow(['SENTIMENT SCORES OVER TIME'])
        writer.writerow(['Timestamp', 'Sentiment Score'])
        sentiment_scores = call_doc.get('sentiment_scores', [])
        for i, score_data in enumerate(sentiment_scores):
            timestamp = score_data.get('timestamp', i)
            score = score_data.get('score', 0)
            writer.writerow([f"T{timestamp}", f"{score:.2f}"])
        writer.writerow([])  # Empty row
        
        # Key phrases
        writer.writerow(['KEY PHRASES'])
        key_phrases = call_doc.get('key_phrases', {'positive': [], 'negative': []})
        
        writer.writerow(['Positive Phrases'])
        for phrase in key_phrases.get('positive', []):
            writer.writerow([phrase])
        
        writer.writerow(['Negative Phrases'])
        for phrase in key_phrases.get('negative', []):
            writer.writerow([phrase])
        
        # Get CSV content
        csv_content = output.getvalue()
        output.close()
        
        # Create response
        response = app.response_class(
            csv_content,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={call_id}_analysis.csv'}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"CSV export failed for {call_id}: {e}")
        return jsonify({'error': 'CSV export failed'}), 500

@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """Start analysis for uploaded file."""
    try:
        data = request.get_json()
        call_id = data.get('call_id')
        
        if not call_id:
            return jsonify({'error': 'call_id is required'}), 400
        
        # Check if call exists in database
        call_doc = db.calls.find_one({'call_id': call_id})
        if not call_doc:
            return jsonify({'error': 'Call not found'}), 404
        
        # Update status to processing
        db.calls.update_one(
            {'call_id': call_id},
            {'$set': {'status': 'processing', 'updated_at': datetime.now()}}
        )
        
        return jsonify({
            'message': 'Analysis started',
            'call_id': call_id,
            'status': 'processing'
        })
    except Exception as e:
        logger.error(f"Analysis start failed: {e}")
        return jsonify({'error': 'Analysis start failed'}), 500

@app.route('/api/results/<call_id>')
def get_results(call_id):
    """Get analysis results for a call."""
    try:
        call_doc = db.calls.find_one({'call_id': call_id})
        if not call_doc:
            return jsonify({'error': 'Call not found'}), 404
        
        # Return the analysis results
        return jsonify({
            'call_id': call_id,
            'sentiment_scores': call_doc.get('sentiment_scores', []),
            'emotions': call_doc.get('emotions', {}),
            'sale_probability': call_doc.get('sale_probability', 0),
            'key_phrases': call_doc.get('key_phrases', {'positive': [], 'negative': []}),
            'summary': {
                'avg_sentiment': call_doc.get('avg_sentiment', 0),
                'total_duration': call_doc.get('duration', 0),
                'participants': call_doc.get('participants', 2)
            }
        })
    except Exception as e:
        logger.error(f"Results retrieval failed for {call_id}: {e}")
        return jsonify({'error': 'Results retrieval failed'}), 500

@app.route('/api/status/<call_id>')
def get_status(call_id):
    """Get analysis status for a call."""
    try:
        call_doc = db.calls.find_one({'call_id': call_id})
        if not call_doc:
            return jsonify({'error': 'Call not found'}), 404
        
        return jsonify({
            'call_id': call_id,
            'status': call_doc.get('status', 'unknown'),
            'progress': call_doc.get('progress', 0)
        })
    except Exception as e:
        logger.error(f"Status check failed for {call_id}: {e}")
        return jsonify({'error': 'Status check failed'}), 500

@app.route('/api/history')
def get_history():
    """Get call history."""
    try:
        calls = list(db.calls.find(
            {},
            {
                'call_id': 1,
                'filename': 1,
                'timestamp': 1,
                'duration': 1,
                'avg_sentiment': 1,
                'sale_probability': 1,
                'participants': 1,
                'status': 1
            }
        ).sort('timestamp', -1))
        
        # Convert ObjectId to string for JSON serialization
        for call in calls:
            call['_id'] = str(call['_id'])
        
        return jsonify(calls)
    except Exception as e:
        logger.error(f"History retrieval failed: {e}")
        return jsonify({'error': 'History retrieval failed'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    try:
        # Check MongoDB connection
        db.command('ping')
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

def create_html_template():
    """Create enhanced HTML template for the web app."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f7fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .conversation-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .conversation-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }
        .conversation-card:hover {
            transform: translateY(-5px);
        }
        .conversation-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .conversation-id {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .analysis-results {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .btn {
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background: #2980b9;
        }
        .loading {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
        }
        .error {
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .upload-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }
        .upload-section input[type="file"] {
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Call Analysis Dashboard</h1>
            <p>AI-powered sentiment analysis and sale prediction system</p>
        </div>

        <div class="upload-section">
            <h3>Upload Audio File</h3>
            <input type="file" id="audio-upload" accept=".wav,.mp3,.m4a">
            <button class="btn" onclick="uploadAudio()">Analyze Audio</button>
            <div id="upload-status"></div>
        </div>

        <div id="conversation-list" class="conversation-list">
            <div class="loading">Loading conversations...</div>
        </div>

        <div id="analysis-results" class="analysis-results" style="display: none;">
            <h2>Analysis Results</h2>
            <div id="selected-conversation"></div>
            
            <div class="metric-grid" id="metrics"></div>

            <div class="chart-container">
                <h3>Sentiment Timeline</h3>
                <div id="sentiment-chart"></div>
            </div>

            <div class="chart-container">
                <h3>Emotion Distribution</h3>
                <div id="emotion-chart"></div>
            </div>

            <div class="chart-container">
                <h3>Sale Probability</h3>
                <div id="sale-chart"></div>
            </div>

            <div class="chart-container">
                <h3>Conversation Flow</h3>
                <div id="flow-chart"></div>
            </div>

            <div class="chart-container">
                <h3>Feature Importance</h3>
                <div id="feature-chart"></div>
            </div>

            <div style="text-align: center; margin-top: 20px;">
                <button class="btn" onclick="exportResults()">Export Results</button>
                <button class="btn" onclick="generateDashboard()">Download Dashboard</button>
                <button class="btn" onclick="previewDashboard()">Preview Dashboard</button>
            </div>
        </div>
    </div>

    <script>
        let currentConversationId = null;
        let currentAnalysis = null;

        document.addEventListener('DOMContentLoaded', function() {
            loadConversations();
        });

        async function loadConversations() {
            try {
                const response = await fetch('/api/conversations');
                const conversations = await response.json();
                if (conversations.error) throw new Error(conversations.error);
                displayConversations(conversations);
            } catch (error) {
                document.getElementById('conversation-list').innerHTML = 
                    `<div class="error">Error loading conversations: ${error.message}</div>`;
            }
        }

        function displayConversations(conversations) {
            const container = document.getElementById('conversation-list');
            container.innerHTML = conversations.map(conv => `
                <div class="conversation-card" onclick="analyzeConversation('${conv.id}')">
                    <div class="conversation-title">${conv.title}</div>
                    <div class="conversation-id">ID: ${conv.id} | Segments: ${conv.segments}</div>
                </div>
            `).join('');
        }

        async function analyzeConversation(conversationId) {
            currentConversationId = conversationId;
            try {
                document.getElementById('analysis-results').style.display = 'block';
                document.getElementById('selected-conversation').innerHTML = 
                    '<div class="loading">Analyzing conversation...</div>';

                const response = await fetch(`/api/analyze/${conversationId}`);
                const analysis = await response.json();
                if (analysis.error) throw new Error(analysis.error);
                
                currentAnalysis = analysis;
                displayAnalysis(analysis);
            } catch (error) {
                document.getElementById('selected-conversation').innerHTML = 
                    `<div class="error">Error analyzing conversation: ${error.message}</div>`;
            }
        }

        async function uploadAudio() {
            const fileInput = document.getElementById('audio-upload');
            const statusDiv = document.getElementById('upload-status');
            if (!fileInput.files.length) {
                statusDiv.innerHTML = '<div class="error">Please select a file</div>';
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                statusDiv.innerHTML = '<div class="loading">Uploading and analyzing...</div>';
                const response = await fetch('/api/upload', { method: 'POST', body: formData });
                const result = await response.json();
                if (result.error) throw new Error(result.error);
                
                currentConversationId = result.analysis.conversation_id;
                currentAnalysis = result.analysis;
                statusDiv.innerHTML = '<div style="color: green;">Analysis complete!</div>';
                displayAnalysis(result.analysis);
            } catch (error) {
                statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
            }
        }

        function displayAnalysis(analysis) {
            document.getElementById('selected-conversation').innerHTML = `
                <h3>${analysis.conversation_title || 'Conversation Analysis'}</h3>
                <p><strong>ID:</strong> ${analysis.conversation_id} | 
                   <strong>Duration:</strong> ${analysis.duration.toFixed(1)}s | 
                   <strong>Segments:</strong> ${analysis.segments}</p>
            `;

            const metrics = analysis.conversation_metrics;
            const salePred = analysis.sale_prediction;
            document.getElementById('metrics').innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${(salePred.sale_probability * 100).toFixed(1)}%</div>
                    <div class="metric-label">Sale Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.average_sentiment.toFixed(2)}</div>
                    <div class="metric-label">Avg Sentiment</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.dominant_emotion}</div>
                    <div class="metric-label">Dominant Emotion</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.sentiment_trend.toFixed(3)}</div>
                    <div class="metric-label">Sentiment Trend</div>
                </div>
            `;

            createSentimentChart(analysis.sentiment_analysis);
            createEmotionChart(analysis.emotion_analysis);
            createSaleChart(analysis.sale_prediction);
            createFlowChart(analysis.segments);
            createFeatureChart(analysis.sale_prediction.feature_importance || {});
        }

        function createSentimentChart(sentimentData) {
            const times = sentimentData.map(s => s.start_time);
            const scores = sentimentData.map(s => s.score);
            const sentiments = sentimentData.map(s => s.sentiment);
            const speakers = sentimentData.map(s => s.speaker);
            const texts = sentimentData.map(s => s.text);

            const trace = {
                x: times,
                y: scores,
                mode: 'lines+markers',
                type: 'scatter',
                name: 'Sentiment Score',
                line: { color: 'blue', width: 3 },
                marker: { size: 8 },
                text: texts,
                hovertemplate: '<b>Time:</b> %{x}s<br><b>Sentiment:</b> %{y:.2f}<br><b>Text:</b> %{text}<br><b>Speaker:</b> ${speakers}<extra></extra>'
            };

            const layout = {
                title: 'Sentiment Timeline',
                xaxis: { title: 'Time (seconds)' },
                yaxis: { title: 'Sentiment Score' },
                height: 400
            };

            Plotly.newPlot('sentiment-chart', [trace], layout);
        }

        function createEmotionChart(emotionData) {
            const emotions = emotionData.map(e => e.emotion);
            const emotionCounts = {};
            emotions.forEach(emotion => {
                emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
            });

            const trace = {
                labels: Object.keys(emotionCounts),
                values: Object.values(emotionCounts),
                type: 'pie',
                hole: 0.3,
                textinfo: 'label+percent',
                textfont: { size: 12 }
            };

            const layout = {
                title: 'Emotion Distribution',
                height: 400
            };

            Plotly.newPlot('emotion-chart', [trace], layout);
        }

        function createSaleChart(salePrediction) {
            const probability = salePrediction.sale_probability * 100;
            const confidence = salePrediction.confidence * 100;
            const color = probability > 70 ? 'green' : probability > 40 ? 'orange' : 'red';

            const trace = {
                type: 'indicator',
                mode: 'gauge+number+delta',
                value: probability,
                domain: { x: [0, 1], y: [0, 1] },
                title: { text: `Sale Probability (%)<br>Confidence: ${confidence.toFixed(1)}%` },
                delta: { reference: 50 },
                gauge: {
                    axis: { range: [null, 100] },
                    bar: { color: color },
                    steps: [
                        { range: [0, 30], color: 'lightgray' },
                        { range: [30, 70], color: 'gray' }
                    ],
                    threshold: {
                        line: { color: 'red', width: 4 },
                        thickness: 0.75,
                        value: 50
                    }
                }
            };

            const layout = { height: 400 };
            Plotly.newPlot('sale-chart', [trace], layout);
        }

        function createFlowChart(segments) {
            const times = segments.map(s => (s.start_time + s.end_time) / 2);
            const speakers = segments.map(s => s.speaker);
            const texts = segments.map(s => s.text.length > 30 ? s.text.slice(0, 30) + '...' : s.text);

            const trace = {
                x: times,
                y: Array(segments.length).fill(1),
                type: 'bar',
                text: texts,
                textposition: 'inside',
                marker: {
                    color: speakers.map(s => s.toLowerCase() === 'customer' ? '#FF6B6B' : '#4ECDC4')
                },
                hovertemplate: '<b>%{text}</b><br>Speaker: %{customdata}<extra></extra>',
                customdata: speakers
            };

            const layout = {
                title: 'Conversation Flow',
                xaxis: { title: 'Time (seconds)' },
                yaxis: { title: 'Speaker', showticklabels: false },
                height: 200,
                barmode: 'stack'
            };

            Plotly.newPlot('flow-chart', [trace], layout);
        }

        function createFeatureChart(featureImportance) {
            const importance = featureImportance.importance || [];
            const featureNames = featureImportance.feature_names || Array.from({ length: importance.length }, (_, i) => `Feature_${i}`);
            const sortedIndices = importance.map((_, i) => i).sort((a, b) => importance[b] - importance[a]).slice(0, 10);
            const sortedImportance = sortedIndices.map(i => importance[i]);
            const sortedNames = sortedIndices.map(i => featureNames[i]);

            const trace = {
                x: sortedImportance,
                y: sortedNames,
                type: 'bar',
                orientation: 'h'
            };

            const layout = {
                title: 'Top 10 Most Important Features',
                xaxis: { title: 'Importance Score' },
                yaxis: { title: 'Features' },
                height: 500
            };

            Plotly.newPlot('feature-chart', [trace], layout);
        }

        async function exportResults() {
            if (!currentConversationId) {
                alert('No conversation selected');
                return;
            }
            try {
                const response = await fetch(`/api/export/${currentConversationId}`);
                if (!response.ok) throw new Error((await response.json()).error);
                const blob = window.URL.createObjectURL(await response.blob());
                const a = document.createElement('a');
                a.href = blob;
                a.download = `${currentConversationId}_analysis.json`;
                a.click();
                window.URL.revokeObjectURL(blob);
            } catch (error) {
                alert(`Error exporting results: ${error.message}`);
            }
        }

        async function generateDashboard() {
            if (!currentConversationId) {
                alert('No conversation selected');
                return;
            }
            try {
                const response = await fetch(`/api/dashboard/${currentConversationId}`);
                if (!response.ok) throw new Error((await response.json()).error);
                const blob = window.URL.createObjectURL(await response.blob());
                const a = document.createElement('a');
                a.href = blob;
                a.download = `${currentConversationId}_dashboard.html`;
                a.click();
                window.URL.revokeObjectURL(blob);
            } catch (error) {
                alert(`Error generating dashboard: ${error.message}`);
            }
        }

        async function previewDashboard() {
            if (!currentConversationId) {
                alert('No conversation selected');
                return;
            }
            try {
                const response = await fetch(`/api/dashboard/${currentConversationId}`);
                if (!response.ok) throw new Error((await response.json()).error);
                const html = await response.text();
                const win = window.open('', '_blank');
                win.document.write(html);
            } catch (error) {
                alert(`Error previewing dashboard: ${error.message}`);
            }
        }
    </script>
</body>
</html>
    """
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

if __name__ == '__main__':
    create_html_template()
    print("Starting Call Analysis Web Application...")
    print("Access the dashboard at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)