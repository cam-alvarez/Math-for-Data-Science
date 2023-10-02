# Image Processing Techniques and Transformations

## Table of Contents
1. [Loading, Resizing, and Cropping Images](#loading-resizing-and-cropping-images)
    - [Loading an Image from a URL](#loading-an-image-from-a-url)
    - [Resizing Images](#resizing-images)
    - [Cropping Images with Indexing](#cropping-images-with-indexing)
    - [Combining Loading, Resizing, and Cropping](#combining-loading-resizing-and-cropping)
2. [Image Transformations](#image-transformations)
    - [Loading an Image](#loading-an-image)
    - [Resizing Images](#resizing-images-1)
    - [Rotating Images](#rotating-images)
    - [Flipping Images](#flipping-images)
    - [Combining Transformations](#combining-transformations)

---

## **Loading, Resizing, and Cropping Images**

Real-world applications often require loading images from external sources, resizing them to specific dimensions, and cropping parts of interest. These essential image manipulation techniques are performed using Python libraries like **`skimage`** and **`matplotlib`**.

### **Loading an Image from a URL**

Using **`skimage`**, loading an image from a URL is straightforward with the **`imread`** function, which reads images from local paths and URLs.

```python
from skimage.io import imread
import matplotlib.pyplot as plt

url = 'https://tinyurl.com/RGBImageCat'
im = imread(url)

plt.figure(figsize=(20,10))
plt.imshow(im)
plt.axis('off')
plt.show()

```

### **Resizing Images**

Resizing is necessary to standardize dimensions. Utilize the **`skimage.transform.resize`** method to resize images.

```python
from skimage.transform import resize

# Resizing the image to 100 x100 pixels
resized_image = resize(im, (100, 100))

plt.imshow(resized_image)
plt.axis('off')
plt.show()
```

### **Cropping Images with Indexing**

Cropping is performed by selecting a region of interest using indexing.

```python
# Cropping the central part of the image
height, width, _ = im.shape
cropped_image = im[height//4:3*height//4, width//4:3*width//4]

plt.imshow(cropped_image)
plt.axis('off')
plt.show()
```

### **Combining Loading, Resizing, and Cropping**

Combine these techniques to process an image effectively.

```python
# Load the image
image = imread(url)
# Resize it
resized_image = resize(image, (200, 200))
# Crop the central part
cropped_image = resized_image[50:150, 50:150]

plt.imshow(cropped_image)
plt.axis('off')
plt.show()
```

## **Image Transformations**

Image transformations are crucial for pre-processing, data augmentation, and various applications in computer vision and graphics. Fundamental techniques like resizing, rotating, and flipping images are extensively practiced.

### **Loading an Image**

Load an image using the same method as in previous chapters of the textbook.

```python
from skimage.io import imread

url = 'https://example.com/image.jpg'
image = imread(url)

plt.imshow(image)
plt.axis('off')
plt.show()
```

### **Resizing Images**

Resize images to normalize them to consistent dimensions or reduce computational requirements.

```python
from skimage.transform import resize

resized_image = resize(image, (50, 50))

plt.imshow(resized_image)
plt.axis('off')
plt.show()
```

### **Rotating Images**

Rotation is used for data augmentation or to correct the orientation of an image.

```python
from skimage.transform import rotate

rotated_image = rotate(image, angle=45)

plt.imshow(rotated_image)
plt.axis('off')
plt.show()
```

### **Flipping Images**

Flipping creates reflections of the image, often used for data augmentation.

```python
from skimage.transform import hflip

flipped_image = hflip(image)

plt.imshow(flipped_image)
plt.axis('off')
plt.show()
```

### **Combining Transformations**

Transformations can be combined to create complex effects or extensive data augmentations.

```python
# Combining resize, rotate, and flip
combined_image = hflip(rotate(resize(image, (100, 100)), angle=30))

plt.imshow(combined_image)
plt.axis('off')
plt.show()
```

---
