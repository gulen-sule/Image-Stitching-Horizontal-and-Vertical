import numpy as np
import cv2
from scipy.spatial.distance import cdist
import imutils

class Stitcher:

    def __init__(self, images, name="Concatenated_images"):
        self.isv3 = imutils.is_cv3(or_better=True)
        results = images.copy()
        cs = self.get_histogram(images, 256)
        num = 0
        (b_des1, b_keys1) = (None, None)
        (b_des2, b_keys2) = (None, None)
        b_matches = None
        index = 0
        print("stitched images loading", end="")
        while True:
            best_matched_number = 0
            best_match_ind = 0
            if len(results) == 1:
                img_output = self.get_normalized_histogram(results[0], cs)
                cv2.imwrite(name + ".jpeg", img_output)
                cv2.imshow("Concatenated Images", img_output)
                cv2.waitKey(0)
            for j in range(len(results)):
                print(".", end="")
                if np.array_equal(results[index], results[j]):
                    continue

                (des1, keys1) = self.get_key_points(results[index])
                (des2, keys2) = self.get_key_points(results[j])
                goodMatches, matches, hasGoodMatches = self.match_keypoints(des1, des2)

                if hasGoodMatches:
                    if best_matched_number < len(goodMatches):
                        best_matched_number = len(goodMatches)
                        best_match_ind = j
                        b_matches = matches[:]
                        b_keys2 = keys2[:]
                        b_keys1 = keys1[:]
            if best_matched_number < 4:
                cv2.imwrite('not matched' + str(num) + ".jpeg", results[0])
                cv2.imshow("not matched", results[0])
                cv2.waitKey(0)
                num += 1
                del results[0]
                continue
            warped = self.warpImage(results[index], b_keys1, b_keys2, b_matches)
            warped_key, warped_img, norm_key, norm_img, best_match = self.concat_images(warped, results[best_match_ind])
            if warped_img is None:
                cv2.imwrite('not matched part' + str(num) + ".jpeg", results[0])
                cv2.imshow("not matched part", results[0])
                cv2.waitKey(0)
                num += 1
                del results[0]
                continue
            cv2.imwrite('warped' + str(num) + ".jpeg", warped_img)
            del results[best_match_ind]
            del results[0]
            results.append(warped_img)

            num += 1
            if len(results) == 1:
                img_output = self.get_normalized_histogram(warped_img, cs)
                img_output_cropped = self.cut_borders(img_output)
                cv2.imwrite(name + ".jpeg", img_output)
                cv2.imshow("Concatenated Images", img_output_cropped)
                cv2.waitKey(0)
                break

        cv2.destroyAllWindows()

    def get_normalized_histogram(self, warped_img, cs):
        img_yuv = cv2.cvtColor(warped_img.astype(np.uint8), cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yv = img_yuv[:, :, 0].flatten()
        for index in range(len(img_yv)):
            img_yv[index] = cs[img_yv[index]]

        img_yuv[:, :, 0] = np.reshape(img_yv, img_yuv[:, :, 0].shape)
        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def get_histogram(self, images, bins):
        # array with size of bins, set to zeros
        histogram = np.zeros(bins)

        # loop through pixels and sum up counts of pixels
        for img in images:
            img_yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2YUV)
            for pixel in img[:, :, 0]:
                histogram[pixel] += 1
        cdf = histogram.cumsum()
        nj = (cdf - cdf.min()) * 255
        N = cdf.max() - cdf.min()

        # re-normalize the cumsum
        cs = nj / N

        # cast it back to uint8 since we can't use floating point values in images
        cs = cs.astype('uint8')
        return cs

    def get_key_points(self, img):
        # create SIFT feature extractor
        """if self.isv3:
            # detect and extract features from the image
            surf = cv2.xfeatures2D.SURF_create ()
            keypoints, descriptors = surf.detectAndCompute(img, None)
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(img)
            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            descriptors, keypoints = extractor.compute(img, kps)"""
        sift = cv2.xfeatures2d.SIFT_create(edgeThreshold=10, contrastThreshold=0.06)
        # detect features from the image
        keypoints, descriptors = sift.detectAndCompute(img, None)
        return descriptors, keypoints

    def match_keypoints(self, des1, des2, ratio_thresh=0.40, ):
        print(".", end="")
        hasGoodMatches = False
        result = (cdist(des1, des2, 'euclidean'))
        matches = []
        for i in range(len(result[:, 0])):
            row = result[i, :]
            min_value = np.amin(row)
            best_match_y = np.where(row == min_value)
            match_one = cv2.DMatch(i, best_match_y[0][0], min_value)
            row = np.delete(row, best_match_y)
            min_value = np.amin(row)
            best_match_y = np.where(row == min_value)
            match_two = cv2.DMatch(i, best_match_y[0][0], min_value)
            best_two_match = [match_one, match_two]
            matches.append(best_two_match)
        """flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        matches = flann.knnMatch(des1, des2, k=2)"""
        goodMatches = []
        matches_im1 = []
        for m1, m2 in matches:
            # ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if m1.distance < ratio_thresh * m2.distance:
                goodMatches.append((m1, m2))
                matches_im1.append(m1)
        if len(goodMatches) > 4:
            hasGoodMatches = True
        return goodMatches, matches_im1, hasGoodMatches

    def draw_matches(self, keys1, img1, keys2, img2, matches):
        img3 = cv2.drawMatchesKnn(img1, keys1, img2, keys2, matches, None, flags=2)
        cv2.imshow("matches", img3)
        cv2.waitKey(0)

    def warpImage(self, img, keys1, keys2, goodMatches, MIN_MATCH_COUNT=4):
        if len(goodMatches) > MIN_MATCH_COUNT:
            src_pts = np.float32([keys1[m.queryIdx].pt for m in goodMatches])
            dst_pts = np.float32([keys2[m.trainIdx].pt for m in goodMatches])

            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
            if H is None:
                H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return img
            height, width = img.shape[:2]
            corners = np.array([
                [0, 0],
                [0, height - 1],
                [width - 1, height - 1],
                [width - 1, 0]
            ])
            corners = cv2.perspectiveTransform(np.float32([corners]), H)[0]
            bx, by, bwidth, bheight = cv2.boundingRect(corners)

            warped = cv2.warpPerspective(img, H, (bwidth, bheight), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT)
            return warped

    def concat_images(self, warped_img, norm_img):
        """"yeniden keypoint bul ve best keypointe onu da distanece değerinin en düşüğünü alarak bulabilirsin
         göre imagei kırpıp birleştir
        """
        warped_des, warped_key = self.get_key_points(warped_img)
        norm_des, norm_key = self.get_key_points(norm_img)
        goodMatches, matches, hasGoodMatches = self.match_keypoints(warped_des, norm_des)
        if len(goodMatches) == 0:
            return warped_key, None, norm_key, norm_img, None
        best_match = goodMatches[0]
        for i in range(len(matches)):
            if best_match[0].distance > matches[i].distance:
                best_match = goodMatches[i]
        # self.draw_matches(warped_key,warped_img,norm_key,norm_img,goodMatches)
        r_warp, c_warp, temp = warped_img.shape
        r_norm, c_norm, temp = norm_img.shape
        c_key1, r_key1 = warped_key[best_match[0].queryIdx].pt
        c_key2, r_key2 = norm_key[best_match[0].trainIdx].pt

        r_key1 = round(r_key1)
        c_key1 = round(c_key1)
        r_key2 = round(r_key2)
        c_key2 = round(c_key2)

        west = (c_key1 if c_key1 > c_key2 else c_key2)
        north = (r_key1 if (r_key1 > r_key2) else r_key2)
        east = ((c_warp - c_key1) if (c_warp - c_key1) > (c_norm - c_key2) else (c_norm - c_key2))
        south = (r_warp - r_key1 if r_warp - r_key1 > r_norm - r_key2 else r_norm - r_key2)
        stitched_img = np.zeros((north + south, east + west, 3), dtype=int)

        warped_loc_r = north - r_key1
        warped_loc_c = west - c_key1

        for a in range(0, r_warp):
            for b in range(0, c_warp):
                stitched_img[warped_loc_r + a][warped_loc_c + b] = warped_img[a, b]

        norm_loc_r = north - r_key2
        norm_loc_c = west - c_key2
        for i in range(r_norm):
            for j in range(c_norm):
                if not np.array_equal(norm_img[i, j], [0, 0, 0]):
                    if np.array_equal(stitched_img[norm_loc_r + i][norm_loc_c + j], [0, 0, 0]):
                        stitched_img[norm_loc_r + i][norm_loc_c + j] = norm_img[i, j]
                    else:
                        stitched_img[norm_loc_r + i][norm_loc_c + j] = norm_img[i, j] * 0.1 + \
                                                                       stitched_img[norm_loc_r + i][
                                                                           norm_loc_c + j] * 0.9

        return warped_key, stitched_img.astype(np.uint8), norm_key, norm_img, best_match

    def cut_borders(self, image):
        left = 0
        right = 0
        top = 0
        bottom = 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r, c = gray.shape
        for i in range(c):
            if sum(gray[:, i]) == 0:
                left += 1
            else:
                break

        for i in range(c - 1, 0, -1):
            if sum(gray[:, i]) == 0:
                right += 1
            else:
                break

        for i in range(r):
            if sum(gray[i, :]) == 0:
                top += 1
            else:
                break

        for i in range(r - 1, 0, -1):
            if sum(gray[i, :]) == 0:
                bottom += 1
            else:
                break

        new_image = np.copy(image[top:(r - bottom) + 1, left:(c - right) + 1, :])
        return new_image
