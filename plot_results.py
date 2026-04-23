import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import os

# File Paths
pred_path = 'results/inference/predictions.csv'
true_path = 'annotations/test_annotations.csv'

if not os.path.exists(pred_path) or not os.path.exists(true_path):
    print("Error: Files not found. Check your paths!")
else:
    # Load data
    preds = pd.read_csv(pred_path)
    true = pd.read_csv(true_path)

    # Merge them
    combined = pd.merge(preds, true, on='ID')

    # USE CHRONOLOGICAL AGE AS THE TRUTH
    target_col = 'chronological_age' 
    
    # Calculate Error
    combined['error'] = abs(combined['predicted_bone_age'] - combined[target_col])
    mae = combined['error'].mean()

    # Create the Comparison Plot
    plt.figure(figsize=(12, 7))
    plt.scatter(combined.index, combined[target_col], color='red', label='Real Age (Ground Truth)', alpha=0.6)
    plt.scatter(combined.index, combined['predicted_bone_age'], color='blue', label='AI Prediction', alpha=0.6)

    # Draw lines between them
    for i in range(len(combined)):
        plt.plot([i, i], [combined[target_col][i], combined['predicted_bone_age'][i]], color='gray', linestyle='--', alpha=0.2)

    plt.title(f'AI vs. Real Age (MAE: {mae:.2f} months)')
    plt.xlabel('Test Image Index')
    plt.ylabel('Age in Months')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.savefig('visualization/final_science_comparison.png')
    print(f"Success! The NEW Average Error is {mae:.2f} months.")
    plt.show()