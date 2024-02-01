import cv2
import imutils  # From PyimageSearch
import numpy as np


class ImageAlignment:
    def __init__(
        self,
        image,
        template,
        maxFeatures=500,
        keepFeatures=0.2,
        method="orb",
        debug=False,
    ):
        self.image = image
        self.template = template
        self.maxFeatures = maxFeatures
        self.keepFeatures = keepFeatures
        self.method = method
        self.debug = debug

    def _preprocess(self):
        imageGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        templateGray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        return imageGray, templateGray

    def _alignmentmethod(self):
        if self.method == "orb":
            align_method = cv2.ORB_create(self.maxFeatures)

        elif self.method == "sift":
            align_method = cv2.SIFT_create(self.maxFeatures)

        elif self.method == "surf":
            align_method = cv2.xfeatures2D.surf_create()

        elif self.method == "akaze":
            align_method = cv2.AKAZE_create()

        elif self.method == "brisk":
            align_method = cv2.BRISK_create()
        return align_method

    def get_keyPoints(self):
        method = self._alignmentmethod()
        imageGray, templateGray = self._preprocess()

        (kpsA, descsA) = method.detectAndCompute(imageGray, None)
        (kpsB, descsB) = method.detectAndCompute(templateGray, None)

        matcher = cv2.DescriptorMatcher_create(
            cv2.NORM_L1
        )  # cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matches = matcher.match(descsA, descsB, None)
        # sort the matches by their distance
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        keep = int(len(matches) * self.keepFeatures)
        matches = matches[:keep]
        # save the key points (x, y)-coordinates from the
        # top matches  use these coordinates to compute the homography matrix
        ptsA = np.zeros((len(matches), 2), dtype="float")
        ptsB = np.zeros((len(matches), 2), dtype="float")
        # loop over the top matches
        for i, m in enumerate(matches):
            # indicate that the two key points in the respective images
            # map to each other
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt

        if self.debug:
            img_kp = cv2.drawKeypoints(
                self.image, kpsA, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.imshow("Keypoints", img_kp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return ptsA, ptsB

    def align_image(self):
        ptA, ptB = self.get_keyPoints()
        (H, _) = cv2.findHomography(ptA, ptB, method=cv2.RANSAC)
        (h, w) = self.template.shape[:2]
        aligned = cv2.warpPerspective(self.image, H, (w, h))
        # return the aligned image
        return aligned

    # def align_image(self):
    #     ptA, ptB = self.get_keyPoints()
    #     return self._calculate_homography(ptA, ptB)
