import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssm
import imutils
import sys
from imagealignment.imagealignment import ImageAlignment


def align_images(
    image1, image2, features=200, keep=0.2, method="orb", already_aligned=False
):
    # Align the images - if not already same size and aligned
    if not already_aligned:
        aligner = ImageAlignment(
            image1,
            image2,
            maxFeatures=features,
            keepFeatures=keep,
            method=method,
            debug=True,
        )
    return aligner.align_image()


def compare_images(
    image1, image2, features=200, keep=0.2, method="orb", already_aligned=False
):
    # Align the images - if not already same size and aligned
    if not already_aligned:
        aligner = ImageAlignment(
            image1,
            image2,
            maxFeatures=features,
            keepFeatures=keep,
            method=method,
            debug=False,
        )
        aligned = aligner.align_image()
    else:
        aligned = image1
    ## Resize the aligned and image2 to the same size
    aligned = imutils.resize(aligned, width=1000)
    image2 = imutils.resize(image2, width=1000)
    # Convert aligned and image2 to grayscale
    gray1 = cv2.cvtColor(aligned.copy(), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2GRAY)

    # compute the strucutural similarity matrix (SSM) on the grayscaled images
    (score, diff) = ssm(gray1, gray2, full=True)
    print(f"Image similarity score = {score}")
    print(f"diff dimensions {diff.shape}")
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print(f"Contour count {len(cnts)}")
    image2_filled = image2.copy()
    i = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 60:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image2_filled, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(image2_filled, [c], 0, (255, 0, 0), 2)
            cv2.putText(
                image2_filled,
                str(i + 1),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 0, 255),
                2,
            )
            i += 1
    cv2.putText(
        image2_filled,
        f"Found {i-1} differences between two images.",
        (30, 40),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2,
    )
    return aligned, image2_filled


def stack_horizontal(im1, im2, title="", margin=None):
    if im1.shape[1] != im2.shape[1]:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

    if margin:
        padding = np.ones((im1.shape[0], int(margin), 3), dtype="uint8")
        padding = padding * 255
        hstacked = np.hstack([im1, padding, im2])
    else:
        hstacked = np.hstack([im1, im2])
    return hstacked


def stack_vertical(im1, im2, margin=None):
    if im1.shape[0] != im2.shape[0]:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

    if margin:
        padding = np.ones((int(margin), im1.shape[1], 3), dtype="uint8")
        padding = padding * 255
        # print(im1.shape[0], im1.shape[1])
        # print(im2.shape[0], im2.shape[1])

        vstacked = np.vstack([im1, padding, im2])
    else:
        vstacked = np.vstack([im1, im2])
    return vstacked


def display_results(im1, im2, title="", hstack=True, margin=20):
    if hstack:
        stacked = stack_horizontal(im1, im2, title=title, margin=margin)
    else:
        stacked = stack_vertical(im1, im2, margin=margin)

    while True:
        cv2.imshow(title, stacked)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
