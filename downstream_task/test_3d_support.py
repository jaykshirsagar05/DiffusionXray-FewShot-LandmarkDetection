#!/usr/bin/env python
"""
Simple test script to verify 3D support in downstream_task module.
This test checks that all necessary components are properly defined.
"""

import sys
import ast
import os

# Add module to path
sys.path.insert(0, '/home/runner/work/DiffusionXray-FewShot-LandmarkDetection/DiffusionXray-FewShot-LandmarkDetection/downstream_task')

def test_3d_code_additions():
    """Test that 3D code additions are present in source files."""
    print("Testing 3D code additions...")
    
    base_path = '/home/runner/work/DiffusionXray-FewShot-LandmarkDetection/DiffusionXray-FewShot-LandmarkDetection/downstream_task'
    
    # Check models.py for 3D additions
    with open(os.path.join(base_path, 'model/models.py'), 'r') as f:
        models_content = f.read()
    
    assert 'WeightStandardizedConv3d' in models_content, "WeightStandardizedConv3d not found in models.py"
    assert 'is_3d' in models_content, "is_3d parameter not found in models.py"
    print("✓ models.py contains 3D additions")
    
    # Check utilities.py for 3D functions
    with open(os.path.join(base_path, 'utilities.py'), 'r') as f:
        utilities_content = f.read()
    
    assert 'points_to_heatmap_3d' in utilities_content, "points_to_heatmap_3d not found"
    assert 'extract_landmarks_3d' in utilities_content, "extract_landmarks_3d not found"
    assert 'generate_heatmap_from_points_3d' in utilities_content, "generate_heatmap_from_points_3d not found"
    print("✓ utilities.py contains 3D functions")
    
    # Check landmarks_datasets.py for Volume3D
    with open(os.path.join(base_path, 'landmarks_datasets.py'), 'r') as f:
        datasets_content = f.read()
    
    assert 'Volume3D' in datasets_content, "Volume3D class not found"
    assert 'nibabel' in datasets_content, "nibabel import not found"
    print("✓ landmarks_datasets.py contains Volume3D class")
    
    # Check main.py for IS_3D parameter
    with open(os.path.join(base_path, 'main.py'), 'r') as f:
        main_content = f.read()
    
    assert 'IS_3D' in main_content, "IS_3D not found in main.py"
    assert 'volume3d' in main_content, "volume3d dataset not found in main.py"
    print("✓ main.py contains 3D support")
    
    print("✓ All 3D code additions test passed!\n")

def test_config_files():
    """Test that configuration files exist."""
    print("Testing configuration files...")
    
    base_path = '/home/runner/work/DiffusionXray-FewShot-LandmarkDetection/DiffusionXray-FewShot-LandmarkDetection/downstream_task'
    
    config_3d_path = os.path.join(base_path, 'config', 'config_3d.json')
    assert os.path.exists(config_3d_path), "config_3d.json not found"
    print("✓ config_3d.json exists")
    
    # Check if is_3d is set to true
    import json
    with open(config_3d_path, 'r') as f:
        config = json.load(f)
    assert config.get('is_3d') == True, "is_3d not set to true in config_3d.json"
    print("✓ is_3d flag is set to true")
    
    usage_doc_path = os.path.join(base_path, '3D_USAGE.md')
    assert os.path.exists(usage_doc_path), "3D_USAGE.md not found"
    print("✓ 3D_USAGE.md exists")
    
    print("✓ Configuration files test passed!\n")

def test_code_structure():
    """Test code structure and syntax."""
    print("Testing code structure...")
    
    base_path = '/home/runner/work/DiffusionXray-FewShot-LandmarkDetection/DiffusionXray-FewShot-LandmarkDetection/downstream_task'
    
    # Test that all files have correct syntax
    files_to_test = [
        'landmarks_datasets.py',
        'utilities.py',
        'model/models.py',
        'main.py'
    ]
    
    for file in files_to_test:
        file_path = os.path.join(base_path, file)
        with open(file_path, 'r') as f:
            code = f.read()
        try:
            ast.parse(code)
            print(f"✓ {file} has valid syntax")
        except SyntaxError as e:
            raise Exception(f"Syntax error in {file}: {e}")
    
    print("✓ Code structure test passed!\n")

def test_backward_compatibility():
    """Test that backward compatibility is maintained."""
    print("Testing backward compatibility...")
    
    base_path = '/home/runner/work/DiffusionXray-FewShot-LandmarkDetection/DiffusionXray-FewShot-LandmarkDetection/downstream_task'
    
    # Check that 2D datasets still exist
    with open(os.path.join(base_path, 'landmarks_datasets.py'), 'r') as f:
        datasets_content = f.read()
    
    assert 'class Chest' in datasets_content, "Chest dataset class not found"
    assert 'class Hand' in datasets_content, "Hand dataset class not found"
    assert 'class Cephalo' in datasets_content, "Cephalo dataset class not found"
    print("✓ 2D dataset classes still present")
    
    # Check that 2D utilities still exist
    with open(os.path.join(base_path, 'utilities.py'), 'r') as f:
        utilities_content = f.read()
    
    assert 'def points_to_heatmap(' in utilities_content, "2D points_to_heatmap not found"
    assert 'def extract_landmarks(' in utilities_content, "2D extract_landmarks not found"
    print("✓ 2D utility functions still present")
    
    # Check that regular config still exists
    config_path = os.path.join(base_path, 'config', 'config.json')
    assert os.path.exists(config_path), "Original config.json not found"
    print("✓ Original config.json still exists")
    
    print("✓ Backward compatibility test passed!\n")

def main():
    """Run all tests."""
    print("="*60)
    print("Running 3D Support Tests for downstream_task Module")
    print("="*60 + "\n")
    
    try:
        test_3d_code_additions()
        test_config_files()
        test_code_structure()
        test_backward_compatibility()
        
        print("="*60)
        print("✓ All static tests passed successfully!")
        print("  Note: Runtime tests require PyTorch and dependencies.")
        print("="*60)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
