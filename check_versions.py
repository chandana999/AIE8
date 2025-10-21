#!/usr/bin/env python3
"""
Version checker script for LogAnalyzer project
"""

import sys
import subprocess
import os

def check_python_version():
    print("🐍 Python Version:")
    print(f"   {sys.version}")
    print()

def check_pip_packages():
    print("📦 Key Package Versions:")
    packages = [
        'fastapi',
        'uvicorn', 
        'langchain',
        'langchain-community',
        'langchain-openai',
        'langgraph',
        'qdrant-client'
    ]
    
    for package in packages:
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'show', package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                version = next((line.split(': ')[1] for line in lines if line.startswith('Version:')), 'Unknown')
                print(f"   {package}: {version}")
            else:
                print(f"   {package}: Not installed")
        except Exception as e:
            print(f"   {package}: Error checking - {e}")
    print()

def check_environment():
    print("🔑 Environment Variables:")
    required_vars = ['OPENAI_API_KEY', 'TAVILY_API_KEY']
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show only first 10 characters for security
            masked_value = value[:10] + "..." if len(value) > 10 else value
            print(f"   {var}: {masked_value} ✅")
        else:
            print(f"   {var}: Not set ❌")
    print()

def check_backend_status():
    print("🚀 Backend Status:")
    try:
        import requests
        response = requests.get('http://localhost:8000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("   Backend API: Running ✅")
            print(f"   Multi-agent system: {'Initialized ✅' if data.get('multi_agent_system_initialized') else 'Not initialized ❌'}")
            print(f"   RAG system: {'Initialized ✅' if data.get('rag_system_initialized') else 'Not initialized ❌'}")
        else:
            print(f"   Backend API: Error (Status {response.status_code}) ❌")
    except Exception as e:
        print(f"   Backend API: Not running ❌ ({e})")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 LogAnalyzer Version Checker")
    print("=" * 60)
    print()
    
    check_python_version()
    check_pip_packages()
    check_environment()
    check_backend_status()
    
    print("=" * 60)
    print("✅ Version check complete!")
    print("=" * 60)


