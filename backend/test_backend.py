#!/usr/bin/env python3
"""
Test script to check backend dependencies and startup
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import flask
        print("✓ Flask imported successfully")
        
        import pymongo
        print("✓ PyMongo imported successfully")
        
        # Test config import
        from config import Config
        print("✓ Config imported successfully")
        
        # Test call_analysis imports
        from call_analysis.web_app import app
        print("✓ Web app imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        print("\nTesting configuration...")
        from config import Config
        
        print(f"MongoDB URI: {Config.MONGODB_URI}")
        print(f"Database: {Config.MONGODB_DATABASE}")
        print(f"Upload folder: {Config.UPLOAD_FOLDER}")
        print(f"Allowed extensions: {Config.ALLOWED_EXTENSIONS}")
        
        print("✅ Configuration loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    print("Backend Dependency Test")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_config()
    
    if success:
        print("\n🎉 Backend is ready to run!")
        print("Run: python run_web_app.py")
    else:
        print("\n❌ Backend has issues that need to be fixed.")
        sys.exit(1)
