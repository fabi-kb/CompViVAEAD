import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CompViVAEAD.src.utils.synthetic_defects import (
    create_synthetic_dataset_dirs,
    generate_synthetic_dataset,
    verify_synthetic_data_structure,
)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic defect datasets')
    parser.add_argument('--class_names', type=str, nargs='+', required=True,
                        help='Classes to generate synthetic data for (e.g., "metal_nut carpet")')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to MVTec H5 dataset')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of synthetic samples to generate per class/type')
    
    args = parser.parse_args()
    
    create_synthetic_dataset_dirs(args.class_names)
    
    # Generate synthetic data
    for class_name in args.class_names:
        # cutout defects
        generate_synthetic_dataset(
            class_name, 
            args.h5_path, 
            defect_type='cutout_synth', 
            num_samples=args.num_samples
        )
        
        # scratch defects
        generate_synthetic_dataset(
            class_name, 
            args.h5_path, 
            defect_type='scratches_synth', 
            num_samples=args.num_samples
        )
    
    # Verify structure
    verify_synthetic_data_structure(args.class_names)

if __name__ == "__main__":
    main()
