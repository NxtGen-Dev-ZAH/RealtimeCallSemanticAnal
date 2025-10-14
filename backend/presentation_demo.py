#!/usr/bin/env python3
"""
Presentation Demo Script for Stakeholders
"""

import time
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from call_analysis.demo import DemoSystem
from call_analysis.models import ConversationAnalyzer
from call_analysis.dashboard import Dashboard

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{title}")
    print("-" * len(title))

def print_metric(label, value, format_type="default"):
    """Print a formatted metric"""
    if format_type == "percentage":
        formatted_value = f"{value:.1%}"
    elif format_type == "decimal":
        formatted_value = f"{value:.2f}"
    else:
        formatted_value = str(value)
    
    print(f"  {label:<25}: {formatted_value}")

def presentation_demo():
    """Run the stakeholder presentation demo"""
    print_header("CALL ANALYSIS SYSTEM - STAKEHOLDER PRESENTATION")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Presented by: Call Analysis Team")
    
    # Initialize system
    print_section("System Initialization")
    print("Initializing AI-powered call analysis system...")
    time.sleep(1)
    
    demo_system = DemoSystem()
    analyzer = ConversationAnalyzer()
    dashboard = Dashboard()
    
    print("✓ Audio processing models loaded")
    print("✓ Text sentiment analysis ready")
    print("✓ Emotion detection models active")
    print("✓ Sale prediction engine initialized")
    print("✓ Dashboard visualization system ready")
    
    # Show available conversations
    print_section("Demo Conversations Available")
    for i, conv in enumerate(demo_system.demo_conversations, 1):
        print(f"{i}. {conv['title']}")
        print(f"   ID: {conv['id']} | Segments: {len(conv['segments'])}")
    
    # Analyze each conversation
    results = []
    for conv in demo_system.demo_conversations:
        print_section(f"Analysis: {conv['title']}")
        
        # Perform analysis
        result = analyzer.analyze_conversation(segments=conv['segments'])
        results.append(result)
        
        # Display key metrics
        sale_prob = result['sale_prediction']['sale_probability']
        avg_sentiment = result['conversation_metrics']['average_sentiment']
        dominant_emotion = result['conversation_metrics']['dominant_emotion']
        duration = result['duration']
        
        print_metric("Sale Probability", sale_prob, "percentage")
        print_metric("Average Sentiment", avg_sentiment, "decimal")
        print_metric("Dominant Emotion", dominant_emotion)
        print_metric("Conversation Duration", f"{duration:.1f}s")
        print_metric("Number of Segments", result['segments'])
        
        # Show conversation flow
        print("\n  Conversation Flow:")
        for segment in conv['segments'][:3]:  # Show first 3 segments
            speaker = segment['speaker']
            text = segment['text'][:60] + "..." if len(segment['text']) > 60 else segment['text']
            print(f"    {speaker}: {text}")
        if len(conv['segments']) > 3:
            print(f"    ... and {len(conv['segments']) - 3} more segments")
        
        time.sleep(2)  # Pause for effect
    
    # Show agent insights
    print_section("Agent Performance Insights")
    agent_insights = analyzer.get_agent_insights(results)
    
    print_metric("Total Conversations", agent_insights['total_conversations'])
    print_metric("Average Sale Probability", agent_insights['average_sale_probability'], "percentage")
    print_metric("Average Sentiment", agent_insights['average_sentiment'], "decimal")
    print_metric("Success Rate", agent_insights['success_rate'], "percentage")
    print_metric("Performance Trend", agent_insights['performance_trend'])
    
    print("\n  Recommendations:")
    for rec in agent_insights['recommendations']:
        print(f"    • {rec}")
    
    # Show system capabilities
    print_section("System Capabilities Demonstrated")
    capabilities = [
        "✓ Real-time sentiment analysis using BERT/DistilBERT",
        "✓ Audio emotion detection with CNN+LSTM models",
        "✓ Speaker diarization and conversation segmentation",
        "✓ Sale probability prediction with XGBoost",
        "✓ Interactive dashboard with Plotly visualizations",
        "✓ Multi-format export (HTML, JSON, Text)",
        "✓ Web-based API for integration",
        "✓ Batch processing for multiple conversations"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
        time.sleep(0.5)
    
    # Show business value
    print_section("Business Value Proposition")
    value_points = [
        "🎯 Improved Sales Conversion: Predict sale probability in real-time",
        "📊 Enhanced Agent Performance: Data-driven insights and recommendations",
        "💰 Cost Reduction: Automated analysis reduces manual review time",
        "📈 Better Customer Experience: Identify and address concerns early",
        "🔍 Actionable Insights: Visual dashboards for quick decision making",
        "⚡ Real-time Processing: Analyze calls as they happen",
        "🔧 Easy Integration: RESTful API for existing systems",
        "📱 Modern Interface: User-friendly web dashboard"
    ]
    
    for value in value_points:
        print(f"  {value}")
        time.sleep(0.7)
    
    # Show technical advantages
    print_section("Technical Advantages")
    advantages = [
        "• Open-source and cost-effective solution",
        "• Student-friendly implementation",
        "• Modular architecture for easy extension",
        "• Multimodal analysis (audio + text)",
        "• Scalable design for enterprise deployment",
        "• Comprehensive documentation and examples"
    ]
    
    for advantage in advantages:
        print(f"  {advantage}")
    
    # Generate outputs
    print_section("Generating Demo Outputs")
    print("Creating interactive dashboard...")
    dashboard_path = demo_system.create_dashboard(results[0])
    print(f"✓ Dashboard saved: {dashboard_path}")
    
    print("Creating summary report...")
    report_path = demo_system.create_summary_report(results[0])
    print(f"✓ Report saved: {report_path}")
    
    print("Creating comparison analysis...")
    comparison_path = demo_system.create_comparison_report()
    print(f"✓ Comparison saved: {comparison_path}")
    
    # Final summary
    print_section("Presentation Summary")
    print("The Call Analysis System prototype successfully demonstrates:")
    print("  • AI-powered sentiment and emotion analysis")
    print("  • Accurate sale probability prediction")
    print("  • Comprehensive conversation insights")
    print("  • Professional visualization dashboards")
    print("  • Ready-to-deploy web application")
    
    print("\nNext Steps:")
    print("  1. Deploy web application: python run_web_app.py")
    print("  2. Access dashboard at: http://localhost:5000")
    print("  3. Review generated reports in 'output/' directory")
    print("  4. Integrate with existing call center systems")
    
    print_header("PRESENTATION COMPLETED")
    print("Thank you for your attention!")
    print("Questions and feedback are welcome.")

if __name__ == "__main__":
    try:
        presentation_demo()
    except KeyboardInterrupt:
        print("\n\nPresentation interrupted by user.")
    except Exception as e:
        print(f"\n\nError during presentation: {e}")
        sys.exit(1)
