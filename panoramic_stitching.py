import cv2
import matplotlib.pyplot as plt
from dataPath import DATA_PATH
import os

# Define constants
SCENE_DIR = "scene"
OUTPUT_FILENAME = "panorama_result.jpg"


def load_images_from_directory(directory_path):
    """
    Loads all images with '.jpg' extension from a specified directory, sorts them
    for a consistent order, and returns the images as a list.

    :param directory_path: The path of the directory containing '.jpg' images.
    :type directory_path: str
    :return: A list of loaded images represented as numpy arrays.
    :rtype: list
    """
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(".jpg")]
    image_files.sort()  # Sort for consistent order
    images = [cv2.imread(filename) for filename in image_files]
    return images


def stitch_images(images):
    """
    Stitches multiple images into a single panoramic image using OpenCV's stitching API.

    The function takes a list of input images and attempts to stitch them into a
    single seamless panoramic image. The process is handled by OpenCV's
    `cv2.Stitcher` class. It returns a status code indicating the outcome of the
    stitching process and the stitched panorama image if successful.

    :param images: List of input images to be stitched together.
                   Each image must be a valid OpenCV image array.
    :type images: list
    :return: A tuple containing the status of the stitching operation and the
             result of the stitched panorama. The `status` is an integer where
             0 denotes success and other codes represent failure cases.
             `stitched_image` is the resulting panoramic image if the operation
             succeeds, otherwise it could be None.
    :rtype: tuple
    """
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)
    return status, stitched_image


def main():
    """
    Main workflow for creating panoramic images from a set of scene images. This function
    initializes the input image directory, processes the images to create a panoramic
    stitch, and handles the output of the final stitched image or failure status. It
    uses OpenCV's image stitching functionality to achieve this.

    :return: None
    """
    # Set up the scene image directory
    scene_directory = os.path.join(DATA_PATH, "images", SCENE_DIR)
    image_list = load_images_from_directory(scene_directory)

    # Perform stitching
    status, stitched_image = stitch_images(image_list)

    # Check stitching status and handle output
    if status == cv2.Stitcher_OK:
        print("Panoramic image created successfully!")
        cv2.imwrite(OUTPUT_FILENAME, stitched_image)
        cv2.imshow("stitched_image", stitched_image)
        print(f"Panoramic image saved at: {OUTPUT_FILENAME}")
    else:
        print(f"Image stitching failed. Status code: {status}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
