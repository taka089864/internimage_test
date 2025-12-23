#!/usr/bin/env python
"""Test the effect of to_rgb parameter on grayscale images."""

import numpy as np
import cv2
import mmcv
import time

# Create a grayscale-like image (RGB channels all have same values)
print("=== Testing to_rgb effect on grayscale images ===")

# Simulate a grayscale image loaded as RGB
height, width = 512, 512
gray_value = np.random.randint(0, 256, (height, width), dtype=np.uint8)
img_gray_as_rgb = np.stack([gray_value, gray_value, gray_value], axis=-1)

print(f"Original image shape: {img_gray_as_rgb.shape}")
print(f"Sample pixel [0,0]: R={img_gray_as_rgb[0,0,0]}, G={img_gray_as_rgb[0,0,1]}, B={img_gray_as_rgb[0,0,2]}")

# Test normalization with to_rgb=True
mean = np.array([148.019, 148.019, 148.019])
std = np.array([57.244, 57.244, 57.244])

# Test 1: to_rgb=True
img1 = img_gray_as_rgb.copy().astype(np.float32)
result1 = mmcv.imnormalize(img1, mean, std, to_rgb=True)

# Test 2: to_rgb=False  
img2 = img_gray_as_rgb.copy().astype(np.float32)
result2 = mmcv.imnormalize(img2, mean, std, to_rgb=False)

# Compare results
print("\n=== Comparison ===")
print(f"Are results identical? {np.allclose(result1, result2)}")
print(f"Max difference: {np.max(np.abs(result1 - result2))}")

# Performance test
print("\n=== Performance Test ===")
n_iterations = 1000

# Test with to_rgb=True
start = time.time()
for _ in range(n_iterations):
    img = img_gray_as_rgb.copy().astype(np.float32)
    _ = mmcv.imnormalize(img, mean, std, to_rgb=True)
time_with_rgb = time.time() - start

# Test with to_rgb=False
start = time.time()
for _ in range(n_iterations):
    img = img_gray_as_rgb.copy().astype(np.float32)
    _ = mmcv.imnormalize(img, mean, std, to_rgb=False)
time_without_rgb = time.time() - start

print(f"Time with to_rgb=True: {time_with_rgb:.3f} seconds")
print(f"Time with to_rgb=False: {time_without_rgb:.3f} seconds")
print(f"Speed improvement: {(time_with_rgb - time_without_rgb) / time_with_rgb * 100:.1f}%")

# Test BGR vs RGB conversion on grayscale
print("\n=== BGR to RGB conversion effect ===")
img_bgr = img_gray_as_rgb.copy()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"Are BGR and RGB identical for grayscale? {np.array_equal(img_bgr, img_rgb)}")
print(f"Sample BGR pixel: {img_bgr[0,0]}")
print(f"Sample RGB pixel: {img_rgb[0,0]}")

# Test with actual color image
print("\n=== Testing with color image ===")
color_img = np.zeros((10, 10, 3), dtype=np.uint8)
color_img[:,:,0] = 100  # Blue in BGR
color_img[:,:,1] = 150  # Green
color_img[:,:,2] = 200  # Red in BGR

print(f"Original BGR: B={color_img[0,0,0]}, G={color_img[0,0,1]}, R={color_img[0,0,2]}")
rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
print(f"After BGR2RGB: R={rgb_img[0,0,0]}, G={rgb_img[0,0,1]}, B={rgb_img[0,0,2]}")

print("\n=== Conclusion ===")
print("For grayscale images (where R=G=B):")
print("- to_rgb=True and to_rgb=False produce identical results")
print("- to_rgb=False is slightly faster as it skips unnecessary BGR2RGB conversion")
print("- The BGR2RGB conversion has no effect on grayscale images")