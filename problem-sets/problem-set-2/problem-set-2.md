# Problem Set 2

- Load RGB image from URL
- Resize image
- Show grayscale copy
- Convolve with 10 random filters and show filters and features maps for each
- Create .md markdown report and post to github
```  
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from scipy.signal import convolve2d
import torch
import torch.nn.functional as F
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```
## Load RGB image from URL
```
# Define the 'plot' function for displaying images as found in the notebook.
def plot(x, cmap='gray'):
fig, ax = plt.subplots()
im = ax.imshow(x, cmap=cmap)
ax.axis('off')
fig.set_size_inches(5, 5)
plt.show()

# Using a random image URL for demonstration.
image_url = "https://tinyurl.com/SpaceShip1234"
image = io.imread(image_url)

# Display the original image
plot(image, cmap='viridis')
#%% md
## Resizing Image
#%%
# Resizing the image to 150x150 pixels
resized_image = transform.resize(image, (150, 150))

# Display the resized image
plot(resized_image, cmap='viridis')
```
## Grayscale Copy
```
# Converting the image to grayscale
gray_image = color.rgb2gray(resized_image)

# Display the grayscale image
plot(gray_image)
```
## Convolve with 10 random filters and show filters and features maps for each
```
# Generate 10 random 3x3 filters
filters = [np.random.rand(3, 3) for _ in range(10)]

# Apply each filter and show the filter and the feature map
for i, f in enumerate(filters):
# Convolve the grayscale image with the filter
feature_map = convolve2d(gray_image, f, mode='same', boundary='wrap')

    # Plot the filter
    plt.title(f"Filter {i+1}")
    plt.imshow(f, cmap='gray')
    plt.show()
    
    # Plot the feature map
    plt.title(f"Feature Map {i+1}")
    plt.imshow(feature_map, cmap='gray')
    plt.show()
```