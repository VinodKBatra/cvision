import cv2
import sys
from image_utils.my_utils import compare_images, display_results
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="image 1")
ap.add_argument("-t", "--template", required=True, help="Template")
ap.add_argument(
    "-f",
    "--features",
    type=int,
    default=200,
    help="Number of features for alignment algorithms",
)
ap.add_argument(
    "-k",
    "--keep",
    type=float,
    default=0.2,
    help="Fraction of features to keep for matching",
)
ap.add_argument(
    "-m",
    "--method",
    type=str,
    default="orb",
    choices=["orb", "sift", "akaze", "surf", "brisk"],
    help="Alignment algorithm options",
)
args = vars(ap.parse_args())


def solve_puzzle(
    image1, image2, method, features=200, keep=0.20, already_aligned=False
):
    """
    Keyword Arguments:
        features --  (default: {200})
        keep -- Fraction to keep (default: {0.20})
        method -- Algorithm (default: {"orb"})
        already_aligned --  (default: {False})
    """

    # path = "./imagealignment/images/"
    # image1 = cv2.imread(path + args["image"])
    # image2 = cv2.imread(path + args["template"])
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
    path = "./imagealignment/images/"
    im1 = cv2.imread(path + args["image"])
    im2 = cv2.imread(path + args["template"])
    feat = args["features"] if args["features"] else 200
    k = args["keep"] if args["keep"] else 0.2
    method = args["method"] if args["method"] else "orb"
    solve_puzzle(im1, im2, method=method, features=feat, keep=k)
