#!/usr/bin/env python3
"""
Debug script to understand the data structure of pickle files
"""
import pickle
import sys

def analyze_pickle_structure(filepath):
    """Analyze the structure of a pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"=== Analysis of {filepath} ===")
        print(f"Main data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Main keys: {list(data.keys())}")
            
            for key, value in data.items():
                print(f"\n{key}:")
                print(f"  Type: {type(value)}")
                if hasattr(value, '__len__'):
                    print(f"  Length: {len(value)}")
                
                if key == 'metadata':
                    if isinstance(value, list) and len(value) > 0:
                        print(f"  First element type: {type(value[0])}")
                        if hasattr(value[0], 'keys'):
                            print(f"  First element keys: {list(value[0].keys())}")
                            
                            # Check author_pairs structure
                            if 'author_pairs' in value[0]:
                                ap = value[0]['author_pairs']
                                print(f"    author_pairs type: {type(ap)}")
                                if hasattr(ap, '__len__'):
                                    print(f"    author_pairs length: {len(ap)}")
                                if hasattr(ap, '__getitem__') and len(ap) > 0:
                                    print(f"    First author_pairs element: {ap[0]}")
                            
                            # Check paths structure
                            if 'paths' in value[0]:
                                paths = value[0]['paths']
                                print(f"    paths type: {type(paths)}")
                                if hasattr(paths, '__len__'):
                                    print(f"    paths length: {len(paths)}")
                                if hasattr(paths, '__getitem__') and len(paths) > 0:
                                    print(f"    First paths element: {paths[0] if len(str(paths[0])) < 100 else str(paths[0])[:100] + '...'}")
                    
                    elif isinstance(value, dict):
                        print(f"  Dict keys: {list(value.keys())}")
                        if 'author_pairs' in value:
                            ap = value['author_pairs']
                            print(f"    author_pairs type: {type(ap)}")
                            if hasattr(ap, '__len__'):
                                print(f"    author_pairs length: {len(ap)}")
        
        return True
    
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test with one of the pickle files
    bags_dir = '/workspace/MIL/data/processed/bags'
    test_file = f'{bags_dir}/bags_arcface_margin_0.4_50p_random_train.pkl'
    
    success = analyze_pickle_structure(test_file)
    if success:
        print("\n✅ Analysis completed successfully")
    else:
        print("\n❌ Analysis failed")