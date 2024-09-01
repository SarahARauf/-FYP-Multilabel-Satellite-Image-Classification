import pandas as pd
from collections import Counter

# Assuming your .csv file is named 'data.csv'
file_path = 'E:\Downloads\BigEarthNetLabels.csv'

# Load the dataset
df = pd.read_csv(file_path)

# Downsample to a fifth of the original size
downsampled_df = df.sample(frac=0.2, random_state=42)

# Count the occurrences of each class label
class_counts = Counter()
for labels in downsampled_df['class']:
    for label in eval(labels):
        class_counts[label] += 1

# Print the class counts
print("Class Label Counts after Downsampling:")
for label, count in class_counts.items():
    print(f"{label}: {count}")

# Save the downsampled dataset to a new .csv file
downsampled_df.to_csv('downsampled_data.csv', index=False)
