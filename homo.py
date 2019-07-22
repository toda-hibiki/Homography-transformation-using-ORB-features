# coding: UTF-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

im1 = cv2.imread('./2019/0520/ab.png',0) # queryImage
im2 = cv2.imread('./2019/0520/aa.png',0) # trainImage

def alignImages(img1, img2,
                max_pts, good_match_rate, min_match):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#feature-homography

    # [1] ORBを用いて特徴量を検出する
    # Initiate ORB detector
    orb = cv2.ORB_create(max_pts)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # [2] 検出した特徴量の比較をしてマッチングをする
    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:int(len(matches) * good_match_rate)]

    # [3] 十分な特徴量が集まったらそれを使って入力画像を変形する
    if len(good) > min_match:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Find homography
        h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        cv2.imwrite('draw_match.jpg', cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:10], None, flags=2))

        # Use homography
        height, width = img1.shape
        dst_img = cv2.warpPerspective(img2, h, (width, height))
        plt.imshow(dst_img, 'gray'),plt.show()
        cv2.imwrite('kekka-ab.png',dst_img)
        return dst_img, h
    else:
        plt.imshow(img1, 'gray'),plt.show()
        return img1, np.zeros((3, 3))

alignImages(im1,im2,500,0.35,10)

