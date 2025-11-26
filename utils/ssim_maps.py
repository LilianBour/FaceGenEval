import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2

img_name = "764.png"
i1 = "/home/bour231/Desktop/gan_inversion/data_evaluation_inversion/data_test_comparison/CelebA_Hq/aligned/"+img_name
i2 = "/home/bour231/Desktop/gan_inversion/data_evaluation_inversion/data_test_comparison/CelebA_Hq/inversion/pti_e4e_base/inversed/"+img_name

# Load images in color (BGR by default in OpenCV)
img1 = cv2.imread(i1)
img2 = cv2.imread(i2)

# Convert BGR → RGB (matplotlib expects RGB)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Compute SSIM for color images
score, diff = ssim(img1, img2, channel_axis=-1, full=True)
print("SSIM Score:", score)

# Create error map (1 - similarity map)
error_map = 1 - diff

# Show results
fig, axes = plt.subplots(1, figsize=(4, 4))
im = axes.imshow(error_map, cmap='jet')
axes.set_title("SSIM Error Map")
axes.axis("off")
fig.colorbar(im, ax=axes)
plt.show()
