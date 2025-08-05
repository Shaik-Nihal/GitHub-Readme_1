#!/usr/bin/env python3
"""
GitHub README Generator Startup Script
This script helps you start the README generator easily.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['flask', 'requests', 'PyGithub', 'python-dotenv', 'PyYAML']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.lower())
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies."""
    print("Installing missing dependencies...")
    for package in missing_packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_env_file():
    """Check if .env file exists and has required keys."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with your API keys.")
        print("Use .env.example as a template.")
        return False
    
    # Read .env file and check for required keys
    with open(env_file, 'r') as f:
        content = f.read()
    
    has_openrouter = 'OPENROUTER_API_KEY=' in content and not content.split('OPENROUTER_API_KEY=')[1].split('\n')[0].strip() == ''
    has_openai = 'OPENAI_API_KEY=' in content and not content.split('OPENAI_API_KEY=')[1].split('\n')[0].strip() == ''
    has_github = 'GITHUB_TOKEN=' in content and not content.split('GITHUB_TOKEN=')[1].split('\n')[0].strip() == ''
    
    if not (has_openrouter or has_openai):
        print("⚠️  No API key found in .env file!")
        print("Please add either OPENROUTER_API_KEY or OPENAI_API_KEY to your .env file.")
        return False
    
    if not has_github:
        print("⚠️  No GitHub token found in .env file!")
        print("Please add GITHUB_TOKEN to your .env file for better rate limits.")
    
    return True

def main():
    """Main function to start the application."""
    print("🚀 GitHub README Generator")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        install_dependencies(missing)
        print("✅ Dependencies installed successfully!")
    
    # Check environment file
    if not check_env_file():
        print("\n💡 You can still use the application by entering API keys in the web interface.")
    
    # Start the Flask application
    print("\n🌐 Starting the web application...")
    print("📍 Open your browser and go to: http://localhost:5001")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Try to open browser automatically
    try:
        webbrowser.open('http://localhost:5001')
    except:
        pass
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("Make sure all dependencies are installed and try again.")

if __name__ == "__main__":
    main()
