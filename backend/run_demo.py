#!/usr/bin/env python3
"""
Run the Call Analysis System Demo
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from call_analysis.demo import main

if __name__ == "__main__":
    print("Starting Call Analysis System Demo...")
    print("=" * 50)
    
    try:
        results = main()
        print("\nDemo completed successfully!")
        print("Check the 'output' directory for generated files:")
        print("- dashboard.html: Interactive dashboard")
        print("- summary_report.txt: Text summary")
        print("- comparison_report.txt: Comparison of all conversations")
        print("- analysis_results.json: Raw analysis data")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)
