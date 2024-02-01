import cv2
import sys
from image_utils.my_utils import compare_images, display_results


def solve_puzzle(features=200, keep=0.20, method="sift", already_aligned=False):
    image1 = cv2.imread("./imagealignment/images/spotthedifference_1.png")
    image2 = cv2.imread("./imagealignment/images/spotthedifference_2.png")
    # Compare the images using compare_images method using default values
    if (image1 is None) or (image2 is None):
        print("Could not read image(s).")
        sys.exit(1)
    else:
        im1, im2 = compare_images(
            image1,
            image2,
            features=features,
            keep=keep,
            method=method,
            already_aligned=already_aligned,
        )

    display_results(image1, image2, title="Original Images", hstack=True, margin=20)
    display_results(
        im1,
        im2,
        title="Results: Spot the differences",
        hstack=True,
        margin=20,
    )
    # To save the results, uncomment the following line
    # cv2.imwrite("stacked_result.png", cv2.hstack([im1, im2]))


if __name__ == "__main__":
    solve_puzzle()
