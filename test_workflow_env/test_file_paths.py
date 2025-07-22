#!/usr/bin/env python3
"""Test file paths used in the workflow scripts"""
import os

def test_file_paths():
    """Check if required paths exist or can be created"""
    
    base_dir = '/home/thierrygc/test_1/github_ready'
    
    paths_to_check = {
        'ChartScanAI weights': os.path.join(base_dir, 'ChartScanAI/weights/custom_yolov8.pt'),
        'Data directory': os.path.join(base_dir, 'data'),
        'Config directory': os.path.join(base_dir, 'config'),
        'NVDA data file': os.path.join(base_dir, 'data/NVDA_15min_pattern_ready.csv'),
        'ChartScanAI_Shiny dir': os.path.join(base_dir, 'ChartScanAI_Shiny'),
        'Evaluation results': os.path.join(base_dir, 'ChartScanAI_Shiny/evaluation_results'),
    }
    
    # Check paths used in download_intraday_data.py
    relative_paths = {
        'directional_analysis': '../directional_analysis',
        'output file': '../directional_analysis/NVDA_intraday_current.csv'
    }
    
    print("Checking absolute paths:")
    print("-" * 50)
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        print(f"{name:<25} {'✓ Exists' if exists else '✗ Not found'}")
        if exists:
            print(f"  Path: {path}")
    
    print("\nChecking relative paths from ChartScanAI_Shiny:")
    print("-" * 50)
    shiny_dir = os.path.join(base_dir, 'ChartScanAI_Shiny')
    for name, rel_path in relative_paths.items():
        abs_path = os.path.abspath(os.path.join(shiny_dir, rel_path))
        exists = os.path.exists(abs_path)
        print(f"{name:<25} {'✓ Exists' if exists else '✗ Not found'}")
        print(f"  Relative: {rel_path}")
        print(f"  Absolute: {abs_path}")
        
        # Check if parent directory exists
        parent_dir = os.path.dirname(abs_path)
        if not exists and os.path.exists(parent_dir):
            print(f"  Parent exists: ✓")

if __name__ == "__main__":
    test_file_paths()