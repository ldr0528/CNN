import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.feature import hog
from skimage import exposure, transform
from skimage.color import gray2rgb
import cv2  # OpenCV is used for CWT visualization

# Define the directories for input and output
input_image_dir = r'D:\HuaweiMoveData\Users\Ldr13\Desktop\SAR\OpenSARship\High speed craft'

# Define the common size to which you want to resize the images
common_size = (256, 256)  # Adjust the size as needed

# Get a list of all JPEG files in the input directory
image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.jpg')]

# Loop through the image files
for image_file in image_files:
    # 1. Read the SAR image
    image = plt.imread(os.path.join(input_image_dir, image_file))  # gray image
    image = gray2rgb(image)

    # 2. Resize the image to the common size
    image = transform.resize(image, common_size)

    # 3. Perform Continuous Wavelet Transform (CWT) on the SAR image
    cwt_result, frequencies = pywt.cwt(image, scales=np.arange(1, 10), wavelet='morl')

    # 4. Select a specific scale from the CWT result (equivalent to time plane in STFT)
    scale_index = 3  # Replace with the desired scale index
    selected_scale = cwt_result[scale_index]

    # 5. Extract HOG_2 features from the selected scale
    fd, hog_image = hog(selected_scale, orientations=6, pixels_per_cell=(18, 18),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)

    # 6. Rescale HOG_2 image intensities for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))

    # 7. Display HOG_2 features and save the result
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    hog_plot = axes.imshow(hog_image_rescaled, cmap='jet')
    axes.axis('off')
    plt.tight_layout()

    result_file = os.path.join(os.path.join('D:\\HuaweiMoveData\\Users\\Ldr13\\Desktop\\SAR\\feature ext\\result2', 'hog_' + image_file))
    plt.savefig(result_file)
    plt.close()
