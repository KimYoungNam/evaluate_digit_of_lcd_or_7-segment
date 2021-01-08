import cv2
import numpy as np
import argparse
from imutils import contours
import imutils

# 실행 시 전달 인자 처리
parser = argparse.ArgumentParser(description='Program to read segment digits')
parser.add_argument('-s', '--show_image', action='store_const', const=True, help='whether to show image')
parser.add_argument('-d', '--debug', action='store_const', const=True, help='whether to show debug message')
parser.add_argument('-a', '--action', required=True, default='radius', choices=['radius', 'weight'], help='what kind of action(radius or weight, default: radius)')
parser.add_argument('-r', '--bw_ratio', required=True, default=0.5, type=float, help='non-zero(ON) pixel ratio(0.0 - 1.0, default: 0.5)')
parser.add_argument('-p', '--path', required=True, default='radius_1.jpg', help='path of capture(image file or camera')
parser.add_argument('-i', '--dilation_iteration_num', required=False, default=5, type=int, help='dilation iteration number(1 - 100, default: 5)')
parser.add_argument('-o', '--threshold_offset', required=False, default=100, type=int, help='simple threshold offset(1 - 255, default: 100)')
# parser.add_argument('--resize_ratio', required=False, default=0.5, help='image resize ratio(range: 0.0 -, default: 0.5)')

# top-down 이미지를 얻기 위해서 네 점을 정렬하기 위한 함수 : four_point_transform()에서 사용
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

# top-down 이미지를 얻기 위한 함수
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

# Digits가 포함된 영역을 찾는 함수 : 원래는 외곽선을 이용해서 해당 영역을 찾아야 하지만 임의로 사각 영역 설정
# 향후 QR 코드 등을 영역에 부착하고 QR 코드의 중심점을 사각 영역의 꼭지점으로 설정할 예정
def find_digits_area(image):
    # digits_area = np.zeros((4, 1, 2))

    # (h, w, c) = image.shape

    # # x1, y1
    # digits_area[0][0][0] = 0
    # digits_area[0][0][1] = 0
    # # x2, y2
    # digits_area[1][0][0] = 0
    # digits_area[1][0][1] = h - 1
    # # x3, y3
    # digits_area[2][0][0] = w - 1
    # digits_area[2][0][1] = 0
    # # x4, y4
    # digits_area[3][0][0] = w - 1
    # digits_area[3][0][1] = h - 1

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red_1 = (0, 100, 100)
    upper_red_1 = (20, 255, 255)

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

    lower_red_2 = (160, 100, 100)
    upper_red_2 = (180, 255, 255)

    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    result = cv2.bitwise_and(image, image, mask=(mask_1 | mask_2))

    # Remove noises
    kernel = np.ones((3, 3), np.int8)
    result = cv2.erode(result, kernel, iterations=2)
    result = cv2.dilate(result, kernel, iterations=2)

    # Fill holes
    result = cv2.dilate(result, kernel, iterations=2)
    result = cv2.erode(result, kernel, iterations=2)
    
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(result)

    digits_area = np.zeros((ret - 1, 1, 2))

    for i in range(1, ret):
        cx, cy = centroids[i]
        digits_area[i - 1][0][0] = cx
        digits_area[i - 1][0][1] = cy

    return digits_area

# Grab image from camera
def get_image(path_to_capture='radius_1.jpg', action='radius', dil_num=5, offset=100, debug=True):
    length_of_digit_area = 0
    if path_to_capture == 'camera':
        image = []
        title = 'Live Frame'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        capture = cv2.VideoCapture(1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
        print(f'width = {capture.get(cv2.CAP_PROP_FRAME_WIDTH)}, height = {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
        while True:
            ret, image = capture.read()
            if ret:
                digits_area = find_digits_area(image)
                length_of_digit_area = len(digits_area)

                if length_of_digit_area == 4:
                    image = four_point_transform(image=image, pts=digits_area.reshape(4, 2))

                    thresholded_image = get_thresholded_image(image=image, which=action, dilation_number=dil_num, threshold_offset=offset, debug_message=False)
                    cv2.imshow(title, cv2.hconcat([image, cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)]))
                    cv2.resizeWindow(title, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.25))
                else:
                    cv2.imshow(title, image)
                    cv2.resizeWindow(title, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5))

                key = cv2.waitKey(1)
                    
                # 'c'apture
                if key == 99 or key == 67:
                    break
                # 's'ave
                elif key == 115 or key == 83:
                    cv2.IMREAD_UNCHANGED
                    cv2.imwrite('image.jpg', image)
                    break
        cv2.destroyAllWindows()
        capture.release()
    else:
        length_of_digit_area = 4
        image = cv2.imread(path_to_capture)

    thresholded_image = get_thresholded_image(image=image, which=action, dilation_number=dil_num, threshold_offset=offset, debug_message=debug)

    return image, thresholded_image, True if length_of_digit_area == 4 else False

# pre-process the image by resizing it, converting it to graycale
def preprocess_image(original_image, resize_ratio):
    resized_color = imutils.resize(original_image, height=int(original_image.shape[0] * resize_ratio))
    resized_gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)
    return resized_color, resized_gray

def get_thresholded_image(image, which='radius', threshold_offset=100, dilation_number=4, debug_message=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    if which == 'radius':
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 111, 5)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 5)
    else:
        # 임계값을 구함 : 임계값 offset(threshold_offset은 실행 시 옵션으로 적용할 것)
        pixel_color_average = np.average(blur.ravel())
        offset = threshold_offset
        threshold_value = pixel_color_average + offset
        if threshold_value > 250:
            threshold_value = 250

        if debug_message:
            print(f'Simple threshold value = {threshold_value}')

        # 전통적인 7-Segment의 경우 주위 조명보다 더 밝은 빛으로 표시되고 꺼져 있는 세그먼트 역시 표시되는 경우가 많다.
        # 이런 이유로 인해 주변 픽셀과 함께 임계값을 동적으로 조정하는 Adaptive 방식은 아래 결과에서 볼 수 있듯이 오동작한다.
        ret, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)

    # Remove noises using Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphology = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Expand segments using Dilating
    kernel = np.ones((3, 3), np.int8)
    dilate = cv2.dilate(morphology, kernel, iterations=dilation_number)

    return dilate

def find_digits(working_image, display_image, debug_message=False):
    cnts = cv2.findContours(working_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method="top-to-bottom")[0]

    height_of_image = working_image.shape[0]

    contour_image = display_image.copy()
    index = 1
    digitCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        message = ''

        if debug_message:
            message = f'Height of image = {height_of_image}, (x, y, w, h) = ({x}, {y}, {w}, {h}), Ratio of height = {h / height_of_image}'

        if h >= (height_of_image * 0.5) and h <= (height_of_image * 0.9):
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(contour_image, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)

            if debug_message:
                message = message + f', and I guess this is a digit(index {index})'
            index = index + 1

            digitCnts.append(c)
        
        if debug_message:
            print(message)
    
    return digitCnts, contour_image

# define the dictionary of digit segments so we can identify each digit
DIGITS_LOOKUP_7_SEGMENT = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

DIGITS_LOOKUP_7x5_SEGMENT = {
    (0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0): 0,
    (0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1): 1,
    (0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1): 2,
    (0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0): 3,
    (0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0): 4,
    (0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0): 5,
    (0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0): 6,
    (0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0): 7,
    (0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0): 8,
    (0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0): 9
}

def get_digit_value(which, digit_cnts, working_image, display_image, on_ratio, debug_message=False):
    digitCnts = contours.sort_contours(digit_cnts, method="left-to-right")[0]
    digits = []
    bw_ratio = on_ratio
    display_image = display_image.copy()
    thresholded_image = working_image.copy()

    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresholded_image[y:y + h, x:x + w]
        # compute the width and height of each of the 7 segments
        (roiH, roiW) = roi.shape

        if which == 'radius':
            # 5 x 7 segments
            (dW, dH) = (int(roiW / 5), int(roiH / 7))
            segments = [
                ((0, 0 * dH), (dW, 1 * dH)),
                ((dW, 0 * dH), (2 * dW, 1 * dH)),
                ((2 * dW, 0 * dH), (3 * dW, 1 * dH)),
                ((3 * dW, 0 * dH), (4 * dW, 1 * dH)),
                ((4 * dW, 0 * dH), (w, 1 * dH)),

                ((0, 1 * dH), (dW, 2 * dH)),
                ((dW, 1 * dH), (2 * dW, 2 * dH)),
                ((2 * dW, 1 * dH), (3 * dW, 2 * dH)),
                ((3 * dW, 1 * dH), (4 * dW, 2 * dH)),
                ((4 * dW, 1 * dH), (w, 2 * dH)),

                ((0, 2 * dH), (dW, 3 * dH)),
                ((dW, 2 * dH), (2 * dW, 3 * dH)),
                ((2 * dW, 2 * dH), (3 * dW, 3 * dH)),
                ((3 * dW, 2 * dH), (4 * dW, 3 * dH)),
                ((4 * dW, 2 * dH), (w, 3 * dH)),

                ((0, 3 * dH), (dW, 4 * dH)),
                ((dW, 3 * dH), (2 * dW, 4 * dH)),
                ((2 * dW, 3 * dH), (3 * dW, 4 * dH)),
                ((3 * dW, 3 * dH), (4 * dW, 4 * dH)),
                ((4 * dW, 3 * dH), (w, 4 * dH)),

                ((0, 4 * dH), (dW, 5 * dH)),
                ((dW, 4 * dH), (2 * dW, 5 * dH)),
                ((2 * dW, 4 * dH), (3 * dW, 5 * dH)),
                ((3 * dW, 4 * dH), (4 * dW, 5 * dH)),
                ((4 * dW, 4 * dH), (w, 5 * dH)),

                ((0, 5 * dH), (dW, 6 * dH)),
                ((dW, 5 * dH), (2 * dW, 6 * dH)),
                ((2 * dW, 5 * dH), (3 * dW, 6 * dH)),
                ((3 * dW, 5 * dH), (4 * dW, 6 * dH)),
                ((4 * dW, 5 * dH), (w, 6 * dH)),

                ((0, 6 * dH), (dW, h)),
                ((dW, 6 * dH), (2 * dW, h)),
                ((2 * dW, 6 * dH), (3 * dW, h)),
                ((3 * dW, 6 * dH), (4 * dW, h)),
                ((4 * dW, 6 * dH), (w, h))
            ]
        else:
            # 7-segments
            (dW, dH) = (int(roiW * 0.3), int(roiH * 0.1))
            segments = [
                ((dW, 0), (w - dW, dH)),  # top
                ((0, dH), (w // 2, h // 2 - (dH // 2))),  # top-left
                ((w // 2, dH), (w, h // 2 - (dH // 2))),  # top-right
                ((dW, (h // 2) - (dH // 2)), (w - dW, (h // 2) + (dH // 2))),  # center
                ((0, h // 2 + (dH // 2)), (w // 2, h - dH)),  # bottom-left
                ((w // 2, h // 2 + (dH // 2)), (w, h - dH)),  # bottom-right
                ((0, h - dH), (w - dW, h))  # bottom
            ]

        on = [0] * len(segments)

        if debug_message:
            print('---------------------')
        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of thresholded pixels in the segment,
            # and then compute the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            
            try:
                if debug_message:
                    print(f'i = {i}, white/black ratio = {total / float(area)}')

                # if the total number of non-zero pixels is greater than 50% of the area, mark the segment as "on"
                if total / float(area) > bw_ratio:
                    on[i] = 1
            except:
                if debug_message:
                    print(f'i = {i}, total = {total}, area = {area}')
                on[i] = 0

        if debug_message:
            print(f'segment = {on}')

        # 폭과 높이의 비율이 일정 비율 미만이면 1로 처리
        if (roiW / roiH) < 0.5:
            digit = 1
        else:
            try:
                if which == 'radius':
                    digit = DIGITS_LOOKUP_7x5_SEGMENT[tuple(on)]
                else:
                    digit = DIGITS_LOOKUP_7_SEGMENT[tuple(on)]
            except:
                digit = '?'
        
        if debug_message:
            print(f'digit = {digit}')

        digits.append(digit)

        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_image, str(digit), (x + w * 1 // 4, y + h * 3 // 4), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)

    return digits, display_image

def main():
    args = parser.parse_args()
    
    while True:
        captured_image, thresholded_image, find_digit_area_is_good = get_image(path_to_capture=args.path, action=args.action, dil_num=args.dilation_iteration_num, offset=args.threshold_offset, debug=args.debug)

        if find_digit_area_is_good:
            if args.show_image:
                captured_title = 'Captured Image'
                cv2.namedWindow(captured_title, cv2.WINDOW_NORMAL)
                cv2.imshow(captured_title, captured_image)
                cv2.resizeWindow(captured_title, (int(captured_image.shape[1] * 1.0), int(captured_image.shape[0] * 1.0)))

            # grayed_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

            # radius default iteration : 5
            # weight default iteration : 11, threshold_offset : 100
            # thresholded_image = get_thresholded_image(image=grayed_image, which=args.action, dilation_number=args.dilation_iteration_num, threshold_offset=args.threshold_offset, debug_message=args.debug)

            if args.show_image:
                threshold_title = 'Thresholded Image'
                cv2.namedWindow(threshold_title, cv2.WINDOW_NORMAL)
                cv2.imshow(threshold_title, thresholded_image)
                cv2.resizeWindow(threshold_title, (int(thresholded_image.shape[1] * 1.0), int(thresholded_image.shape[0] * 1.0)))

            digits, contour_image = find_digits(working_image=thresholded_image, display_image=captured_image, debug_message=args.debug)

            if len(digits) == 0:
                print('There are no digits')
            else:
                if args.show_image:
                    contour_title = 'I guess there are digits'
                    cv2.namedWindow(contour_title, cv2.WINDOW_NORMAL)
                    cv2.imshow(contour_title, contour_image)
                    cv2.resizeWindow(contour_title, (int(contour_image.shape[1] * 1.0), int(contour_image.shape[0] * 1.0)))

                # radius default ratio : 0.7
                # weight default ratio : 0.4
                digits, result_image = get_digit_value(which=args.action, digit_cnts=digits, working_image=thresholded_image, display_image=captured_image, on_ratio=args.bw_ratio, debug_message=args.debug)

                print('===========================')
                print(f'Value of each digit = {digits}')
                print('===========================')

                if args.show_image:
                    result_title = 'I guess these numbers'
                    cv2.namedWindow(result_title, cv2.WINDOW_NORMAL)
                    cv2.imshow(result_title, result_image)
                    cv2.resizeWindow(result_title, (int(result_image.shape[1] * 1.0), int(result_image.shape[0] * 1.0)))

            if args.show_image:
                key = cv2.waitKey(0)
                if key == 113 or key == 81 or key == 27:
                    print('Quit')
                    break
                else:
                    cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()