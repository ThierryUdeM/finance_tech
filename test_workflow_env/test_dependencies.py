#!/usr/bin/env python3
"""Test if all required dependencies can be imported"""

def test_imports():
    """Test importing all required modules"""
    results = {}
    
    # Test imports from requirements.txt
    modules = [
        'yfinance',
        'mplfinance',
        'pandas',
        'ultralytics',
        'azure.storage.blob',
        'dotenv',
        'numpy',
        'matplotlib',
        'PIL',
        'seaborn',
        'scipy'
    ]
    
    for module in modules:
        try:
            if module == 'azure.storage.blob':
                import azure.storage.blob
            elif module == 'dotenv':
                import dotenv
            elif module == 'PIL':
                import PIL
            else:
                __import__(module)
            results[module] = "✓ Installed"
        except ImportError:
            results[module] = "✗ Not installed"
        except Exception as e:
            results[module] = f"✗ Error: {e}"
    
    # Print results
    print("Dependency Check Results:")
    print("-" * 40)
    for module, status in results.items():
        print(f"{module:<20} {status}")
    
    # Check Python version
    import sys
    print(f"\nPython version: {sys.version}")
    
    return all("✓" in status for status in results.values())

if __name__ == "__main__":
    all_installed = test_imports()
    exit(0 if all_installed else 1)