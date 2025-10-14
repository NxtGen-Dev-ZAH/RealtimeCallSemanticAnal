#!/usr/bin/env python3
"""
Run the Call Analysis System Web Application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from call_analysis.web_app import app

if __name__ == "__main__":
    print("Starting Call Analysis Web Application...")
    print("=" * 50)
    print("Access the dashboard at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting web application: {e}")
        sys.exit(1)
