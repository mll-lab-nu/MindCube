#!/usr/bin/env python3
"""
Multiprocess OpenAI API Processing Manager
==========================================

This script manages the multiprocess execution of OpenAI API calls with the following features:
1. Automatic resume from where it left off
2. Progress monitoring
3. Error handling and recovery
4. Results validation

Usage:
    python run_multiprocess.py [--processes NUM] [--validate] [--status]
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openai_other_models import main, check_existing_results, output_path, file_path

def validate_environment():
    """Validate that all required environment variables and files exist."""
    print("🔍 Validating environment...")
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not found!")
        print("   Please set it in your .env file or environment")
        return False
    
    # Check input file
    if not os.path.exists(file_path):
        print(f"❌ Input file not found: {file_path}")
        return False
    
    # Check if input file is readable
    try:
        with open(file_path, 'r') as f:
            first_line = f.readline()
            json.loads(first_line.strip())
        print(f"✅ Input file is valid: {file_path}")
    except Exception as e:
        print(f"❌ Input file is not valid JSON: {e}")
        return False
    
    print("✅ Environment validation passed!")
    return True

def show_status():
    """Show current processing status."""
    print("📊 Current Status")
    print("=" * 50)
    
    # Check input data
    total_items = 0
    try:
        with open(file_path, 'r') as f:
            total_items = sum(1 for line in f)
        print(f"📝 Total items in input: {total_items}")
    except Exception as e:
        print(f"❌ Cannot read input file: {e}")
        return
    
    # Check completed results
    completed_ids = check_existing_results(output_path)
    completed_count = len(completed_ids)
    
    print(f"✅ Completed items: {completed_count}")
    print(f"⏳ Remaining items: {total_items - completed_count}")
    print(f"📈 Progress: {completed_count/total_items*100:.1f}%")
    
    if os.path.exists(output_path):
        stat = os.stat(output_path)
        print(f"📁 Output file size: {stat.st_size / (1024*1024):.2f} MB")
        print(f"🕒 Last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))}")
    
    # Check log file
    log_path = "openai_processing.log"
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"📋 Recent log entries:")
                for line in lines[-5:]:
                    print(f"   {line.strip()}")

def validate_results():
    """Validate the integrity of results file."""
    print("🔍 Validating results...")
    
    if not os.path.exists(output_path):
        print("❌ No results file found")
        return False
    
    valid_count = 0
    invalid_count = 0
    
    try:
        with open(output_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'id' in data and 'answer' in data:
                        valid_count += 1
                    else:
                        print(f"⚠️  Line {line_num}: Missing required fields")
                        invalid_count += 1
                except json.JSONDecodeError:
                    print(f"❌ Line {line_num}: Invalid JSON")
                    invalid_count += 1
        
        print(f"✅ Valid results: {valid_count}")
        print(f"❌ Invalid results: {invalid_count}")
        print(f"📊 Validation rate: {valid_count/(valid_count+invalid_count)*100:.1f}%")
        
        return invalid_count == 0
        
    except Exception as e:
        print(f"❌ Error validating results: {e}")
        return False

def main_runner():
    parser = argparse.ArgumentParser(description='Run multiprocess OpenAI API processing')
    parser.add_argument('--processes', '-p', type=int, default=64, 
                       help='Number of processes to use (default: 64)')
    parser.add_argument('--validate', '-v', action='store_true',
                       help='Validate results file integrity')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Show current processing status')
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    if args.validate:
        validate_results()
        return
    
    # Validate environment before starting
    if not validate_environment():
        print("\n❌ Environment validation failed. Please fix the issues above.")
        sys.exit(1)
    
    # Show initial status
    print("\n" + "="*60)
    print("🚀 Starting Multiprocess OpenAI API Processing")
    print("="*60)
    show_status()
    print("\n")
    
    # Update process count if specified
    if args.processes != 64:
        import openai_other_models
        openai_other_models.NUM_PROCESSES = args.processes
        print(f"🔧 Using {args.processes} processes")
    
    # Start processing
    print("🏃 Starting processing...")
    start_time = time.time()
    
    try:
        main()
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("🎉 Processing completed successfully!")
        print(f"⏱️  Total time: {duration/60:.1f} minutes")
        print("="*60)
        
        # Show final status
        show_status()
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        print("💾 Results have been saved up to the interruption point")
        show_status()
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("💾 Partial results may have been saved")
        show_status()

if __name__ == "__main__":
    main_runner() 