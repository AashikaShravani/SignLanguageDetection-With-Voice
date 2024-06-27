import numpy as np
import pickle

# Assuming you have image data and labels, for example purposes:
# Generate dummy data for demonstration
train_images = np.random.rand(1000, 64, 64)  # 1000 images of 64x64 pixels
train_labels = np.random.randint(0, 10, 1000)  # 1000 labels for 10 classes

test_images = np.random.rand(200, 64, 64)  # 200 images of 64x64 pixels
test_labels = np.random.randint(0, 10, 200)  # 200 labels for 10 classes

# Save the data to files
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)

with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)

print("Dataset files created successfully.")
