#!/usr/bin/env python3
"""
Comprehensive Test Suite for BasicAgent
Covers all major capabilities with real data processing
"""

import os
import sys
import json
import time
import logging
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the agent code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import BasicAgent

# Create sample files for testing
import pandas as pd
import PyPDF2
from PIL import Image
import numpy as np

class TestEnvironment:
    """Helper class to create test files and manage test environment"""
    
    def __init__(self):
        self.test_dir = tempfile.mkdtemp(prefix="agent_test_")
        self.files = {}
        logging.info(f"Created test directory: {self.test_dir}")
        
    def cleanup(self):
        """Clean up test files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logging.info(f"Cleaned up test directory: {self.test_dir}")
    
    def create_text_file(self, filename: str, content: str) -> str:
        """Create a text file with specific content"""
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        self.files[filename] = filepath
        logging.info(f"Created text file: {filename}")
        return filepath
    
    def create_csv_file(self, filename: str, data: Dict) -> str:
        """Create a CSV file with specific data"""
        filepath = os.path.join(self.test_dir, filename)
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        self.files[filename] = filepath
        logging.info(f"Created CSV file: {filename}")
        return filepath
    
    def create_excel_file(self, filename: str, data: Dict, sheet_name: str = 'Sheet1') -> str:
        """Create an Excel file with specific data"""
        filepath = os.path.join(self.test_dir, filename)
        df = pd.DataFrame(data)
        df.to_excel(filepath, index=False, sheet_name=sheet_name)
        self.files[filename] = filepath
        logging.info(f"Created Excel file: {filename}")
        return filepath
    
    def create_pdf_file(self, filename: str, content: str) -> str:
        """Create a simple PDF file (text-based)"""
        filepath = os.path.join(self.test_dir, filename)
        # Create a simple PDF using reportlab if available, otherwise create a text file and rename
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(filepath, pagesize=letter)
            y = 750
            for line in content.split('\n'):
                c.drawString(100, y, line)
                y -= 20
            c.save()
        except ImportError:
            # Fallback: create a text file with .pdf extension
            with open(filepath, 'w') as f:
                f.write(content)
        
        self.files[filename] = filepath
        logging.info(f"Created PDF file: {filename}")
        return filepath
    
    def create_image_file(self, filename: str, text: str) -> str:
        """Create an image with text"""
        filepath = os.path.join(self.test_dir, filename)
        
        # Create image with text
        img = Image.new('RGB', (400, 200), color='white')
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            # Use default font
            draw.text((50, 100), text, fill='black')
        except:
            # If drawing fails, create a simple white image
            pass
        
        img.save(filepath)
        self.files[filename] = filepath
        logging.info(f"Created image file: {filename}")
        return filepath
    
    def create_audio_file(self, filename: str, text: str) -> str:
        """Create an audio file placeholder"""
        # Note: Creating actual audio files requires complex libraries
        # For testing, we'll create a placeholder file
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(b'Audio placeholder for: ' + text.encode())
        self.files[filename] = filepath
        logging.info(f"Created audio placeholder: {filename}")
        return filepath


class ComprehensiveTestSuite:
    """Main test suite for BasicAgent"""
    
    def __init__(self):
        self.env = TestEnvironment()
        self.agent = BasicAgent()
        self.results = []
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.cleanup()
    
    def run_test(self, test_name: str, question: str, files: Dict[str, str], expected: str) -> Dict:
        """Run a single test case"""
        logging.info(f"\n{'='*50}")
        logging.info(f"Running test: {test_name}")
        logging.info(f"Question: {question}")
        logging.info(f"Expected: {expected}")
        
        start_time = time.time()
        
        try:
            # Prepare file metadata if files are provided
            file_metadata = None
            if files:
                file_metadata = {
                    filename: {
                        'filename': filename,
                        'path': filepath,
                        'file_type': os.path.splitext(filename)[1][1:].lower()
                    }
                    for filename, filepath in files.items()
                }
                logging.info(f"Files attached: {list(files.keys())}")
            
            # Run the agent
            result = self.agent(question, file_metadata)
            
            duration = time.time() - start_time
            success = result.strip() == expected.strip()
            
            test_result = {
                'test_name': test_name,
                'success': success,
                'expected': expected,
                'actual': result,
                'duration': duration,
                'error': None
            }
            
            if success:
                logging.info(f"✅ Test PASSED in {duration:.2f}s")
            else:
                logging.error(f"❌ Test FAILED in {duration:.2f}s")
                logging.error(f"Expected: '{expected}'")
                logging.error(f"Actual: '{result}'")
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = {
                'test_name': test_name,
                'success': False,
                'expected': expected,
                'actual': None,
                'duration': duration,
                'error': str(e)
            }
            logging.error(f"❌ Test FAILED with error: {e}")
        
        self.results.append(test_result)
        return test_result
    
    def run_all_tests(self):
        """Run the complete test suite"""
        logging.info("Starting Comprehensive Test Suite")
        
        # Test 1: Text File Processing
        self.env.create_text_file('project_report.txt', '''
Project Griffin Status Report
Date: March 15, 2024
Status: On Track
Budget: $125,750
Completion: 85%

Key Highlights:
- Phase 1 completed successfully
- Phase 2 in progress
- Final delivery scheduled for April 2024
''')
        
        self.run_test(
            'Text File - Date Extraction',
            'What is the date mentioned in the attached project_report.txt?',
            {'project_report.txt': self.env.files['project_report.txt']},
            'March 15, 2024'
        )
        
        # Test 2: CSV Processing with Sum
        self.env.create_csv_file('sales_data.csv', {
            'Product': ['Widget A', 'Widget B', 'Widget C'],
            'Quantity': [100, 200, 150],
            'Price': [10.50, 25.00, 15.75],
            'Total': [1050, 5000, 2362.50]
        })
        
        self.run_test(
            'CSV File - Sum Calculation',
            'What is the sum of the Total column in sales_data.csv? Format as number.',
            {'sales_data.csv': self.env.files['sales_data.csv']},
            '8412.5'
        )
        
        # Test 3: Excel Processing
        self.env.create_excel_file('inventory.xlsx', {
            'Item': ['Laptop', 'Monitor', 'Keyboard', 'Mouse'],
            'Stock': [25, 40, 100, 150],
            'Price': [899.99, 299.99, 79.99, 29.99]
        })
        
        self.run_test(
            'Excel File - Item Count',
            'How many items are listed in inventory.xlsx?',
            {'inventory.xlsx': self.env.files['inventory.xlsx']},
            '4'
        )
        
        # Test 4: Complex Reasoning - Grocery List
        self.env.create_text_file('grocery_list.txt', '''
Shopping list for the week:
1. Apples
2. Bread
3. Milk
4. Tomatoes
5. Bananas
6. Cheese
7. Rice
''')
        
        self.run_test(
            'Complex Reasoning - Fruit Filter',
            'From grocery_list.txt, remove botanical fruits. How many items remain?',
            {'grocery_list.txt': self.env.files['grocery_list.txt']},
            '4'
        )
        
        # Test 5: Web Scraping
        self.run_test(
            'Web Scraping - Simple Page',
            'What is the title of the webpage at example.com?',
            {},
            'Example Domain'
        )
        
        # Test 6: Search Query
        self.run_test(
            'Search Query - Capital',
            'What is the capital of France?',
            {},
            'Paris'
        )
        
        # Test 7: YouTube Title
        self.run_test(
            'YouTube - Video Title',
            'What is the title of the YouTube video at https://www.youtube.com/watch?v=jNQXAC9IVRw?',
            {},
            'Me at the zoo'
        )
        
        # Test 8: PDF Processing
        self.env.create_pdf_file('contract.pdf', '''
Contract Terms and Conditions

Effective Date: January 1, 2024
Contract Number: CTR-2024-001
Client: Acme Corporation
Value: $75,000

This agreement is valid for one year.
''')
        
        self.run_test(
            'PDF File - Contract Value',
            'What is the contract value mentioned in contract.pdf?',
            {'contract.pdf': self.env.files['contract.pdf']},
            '$75,000'
        )
        
        # Test 9: Simple Arithmetic
        self.run_test(
            'Simple Arithmetic',
            'What is 125 + 375?',
            {},
            '500'
        )
        
        # Test 10: Unit Conversion
        self.run_test(
            'Unit Conversion - Temperature',
            'Convert 100 degrees Celsius to Fahrenheit.',
            {},
            '212'
        )
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed
        
        logging.info(f"\n{'='*50}")
        logging.info("TEST SUMMARY")
        logging.info(f"Total tests: {total}")
        logging.info(f"Passed: {passed} ({passed/total*100:.1f}%)")
        logging.info(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        if failed > 0:
            logging.info("\nFAILED TESTS:")
            for r in self.results:
                if not r['success']:
                    logging.error(f"- {r['test_name']}")
                    if r['error']:
                        logging.error(f"  Error: {r['error']}")
                    else:
                        logging.error(f"  Expected: {r['expected']}")
                        logging.error(f"  Actual: {r['actual']}")
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': passed/total*100
        }


def main():
    """Run the comprehensive test suite"""
    with ComprehensiveTestSuite() as test_suite:
        test_suite.run_all_tests()


if __name__ == "__main__":
    main()