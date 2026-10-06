import json
import random

# Load the original train.json
with open('train.json', 'r') as f:
    data = json.load(f)

# Shuffle the data to ensure a random split
random.seed(42) # Set seed for reproducibility
random.shuffle(data)

# Calculate 10% for validation
val_size = int(len(data) * 0.1)

val_data = data[:val_size]
train_data = data[val_size:]

# Save the new val.json
with open('val.json', 'w') as f:
    json.dump(val_data, f, indent=2)

# Overwrite the train.json with the remaining 90%
with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

print(f"Split complete!")
print(f"Original train.json had {len(data)} images.")
print(f"New train.json has {len(train_data)} images.")
print(f"New val.json has {len(val_data)} images.")
