import os

import cv2
import numpy as np

char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
global car_img


def gaussian_blur(image, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny(image, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(image, low_threshold, high_threshold)


def hist_image(image):
    assert image.ndim == 2
    hist = [0 for _ in range(256)]
    img_h, img_w = image.shape[0], image.shape[1]

    for row in range(img_h):
        for col in range(img_w):
            hist[image[row, col]] += 1
    p = [hist[n] / (img_w * img_h) for n in range(256)]
    p1 = np.cumsum(p)
    for row in range(img_h):
        for col in range(img_w):
            v = image[row, col]
            image[row, col] = p1[v] * 255
    return image


def find_board_area(image):
    assert image.ndim == 2
    img_h, img_w = image.shape[0], image.shape[1]
    top, bottom, left, right = 0, img_h, 0, img_w
    flag = False
    h_proj = [0 for _ in range(img_h)]
    v_proj = [0 for _ in range(img_w)]

    for row in range(round(img_h * 0.5), round(img_h * 0.8), 3):
        for col in range(img_w):
            if image[row, col] == 255:
                h_proj[row] += 1
        if not flag and h_proj[row] > 12:
            flag = True
            top = row
        if flag and row > top + 8 and h_proj[row] < 12:
            bottom = row
            flag = False

    for col in range(round(img_w * 0.3), img_w, 1):
        for row in range(top, bottom, 1):
            if image[row, col] == 255:
                v_proj[col] += 1
        if not flag and (v_proj[col] > 10 or v_proj[col] - v_proj[col - 1] > 5):
            left = col
            break
    return left, top, 120, bottom - top - 10


def verify_scale(rotate_rect):
    error = 0.4
    aspect = 4  # 4.7272
    min_area = 10 * (10 * aspect)
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * (1 - error)
    max_aspect = aspect * (1 + error)
    theta = 30

    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False
    ratio = rotate_rect[1][0] / rotate_rect[1][1]
    ratio = max(ratio, 1 / ratio)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if min_area < area < max_area and min_aspect < ratio < max_aspect:
        # check if the angel of rectangle exceeds theta
        if (rotate_rect[1][0] < rotate_rect[1][1] and -90 <= -rotate_rect[2] < -(90 - theta)) \
                or (rotate_rect[1][1] < rotate_rect[1][0] and -theta < -rotate_rect[2] <= 0):
            print('1', area, ratio, rotate_rect[2])
            print('2', min_area, max_area, min_aspect, max_aspect)
            return True
    return False


def img_Transform(car_rect, image):
    global car_img
    img_h, img_w = image.shape[:2]
    rect_w, rect_h = car_rect[1][0], car_rect[1][1]
    angle = car_rect[2]

    return_flag = False
    if car_rect[2] == 0:
        return_flag = True
    if car_rect[2] == -90 and rect_w < rect_h:
        rect_w, rect_h = rect_h, rect_w
        return_flag = True
    if return_flag:
        car_img = image[int(car_rect[0][1] - rect_h / 2):int(car_rect[0][1] + rect_h / 2),
                        int(car_rect[0][0] - rect_w / 2):int(car_rect[0][0] + rect_w / 2)]
        return car_img

    car_rect = (car_rect[0], (rect_w, rect_h), angle)
    box = cv2.boxPoints(car_rect)

    heigth_point = right_point = [0, 0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # positive angel
        new_right_point = [right_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([left_point, heigth_point, new_right_point])
        M = cv2.getAffineTransform(pts1, pts2)  # get affine transform matrix
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))  # apply transformation
        car_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # negative angel
        new_left_point = [left_point[0], heigth_point[1]]
        pts1 = np.float32([left_point, heigth_point, right_point])
        pts2 = np.float32([new_left_point, heigth_point, right_point])
        M = cv2.getAffineTransform(pts1, pts2)  # get affine transform matrix
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))  # apply transformation
        car_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
    return car_img


def pre_process(orig_img):
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_img', gray_img)

    # blur_img = cv2.blur(gray_img, (3, 3))
    kernel_size = 5
    gauss_gray = gaussian_blur(gray_img, kernel_size)  # gaussian blue
    # cv2.imshow('blur', gauss_gray)

    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray, low_threshold, high_threshold)
    # cv2.imshow('canny', canny_edges)

    # sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    # sobel_img = cv2.convertScaleAbs(sobel_img)
    # cv2.imshow('sobel', sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # cv2.imshow('hsv_img', hsv_img)

    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')
    # cv2.imshow('blue_img', blue_img)

    mix_img = np.multiply(canny_edges, blue_img)
    # cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('binary', binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('close', close_img)

    return close_img


def verify_color(rotate_rect, src_image):
    img_h, img_w = src_image.shape[:2]
    mask = np.zeros(shape=[img_h + 2, img_w + 2], dtype=np.uint8)
    connectivity = 4  # [loDiff,upDiff] replaced with new_value, could be 8
    loDiff, upDiff = 30, 30
    new_value = 255
    flags = connectivity
    flags |= cv2.FLOODFILL_FIXED_RANGE
    flags |= new_value << 8
    flags |= cv2.FLOODFILL_MASK_ONLY

    # find column and row range
    rand_seed_num = 5000
    valid_seed_num = 200
    adjust_param = 0.1
    box_points = cv2.boxPoints(rotate_rect)
    box_points_x = [n[0] for n in box_points]
    box_points_x.sort(reverse=False)
    adjust_x = int((box_points_x[2] - box_points_x[1]) * adjust_param)
    col_range = [box_points_x[1] + adjust_x, box_points_x[2] - adjust_x]
    box_points_y = [n[1] for n in box_points]
    box_points_y.sort(reverse=False)
    adjust_y = int((box_points_y[2] - box_points_y[1]) * adjust_param)
    row_range = [box_points_y[1] + adjust_y, box_points_y[2] - adjust_y]

    # rotation adjustment
    if (col_range[1] - col_range[0]) / (box_points_x[3] - box_points_x[0]) < 0.4 \
            or (row_range[1] - row_range[0]) / (box_points_y[3] - box_points_y[0]) < 0.4:
        points_row = []
        points_col = []
        for i in range(2):
            pt1, pt2 = box_points[i], box_points[i + 2]
            x_adjust, y_adjust = int(adjust_param * (abs(pt1[0] - pt2[0]))), int(adjust_param * (abs(pt1[1] - pt2[1])))
            if pt1[0] <= pt2[0]:
                pt1[0], pt2[0] = pt1[0] + x_adjust, pt2[0] - x_adjust
            else:
                pt1[0], pt2[0] = pt1[0] - x_adjust, pt2[0] + x_adjust
            if pt1[1] <= pt2[1]:
                pt1[1], pt2[1] = pt1[1] + adjust_y, pt2[1] - adjust_y
            else:
                pt1[1], pt2[1] = pt1[1] - y_adjust, pt2[1] + y_adjust
            temp_list_x = [int(x) for x in np.linspace(pt1[0], pt2[0], int(rand_seed_num / 2))]
            temp_list_y = [int(y) for y in np.linspace(pt1[1], pt2[1], int(rand_seed_num / 2))]
            points_col.extend(temp_list_x)
            points_row.extend(temp_list_y)
    else:
        points_row = np.random.randint(row_range[0], row_range[1], size=rand_seed_num)
        points_col = np.linspace(col_range[0], col_range[1], num=rand_seed_num).astype(int)

    points_row = np.array(points_row)
    points_col = np.array(points_col)
    hsv_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    flood_img = src_image.copy()  # flood fill image
    seed_cnt = 0
    for i in range(rand_seed_num):
        rand_index = np.random.choice(rand_seed_num, 1, replace=False)
        row, col = points_row[rand_index][0], points_col[rand_index][0]
        # set the color to be car plate color
        if (((h[row, col] > 26) & (h[row, col] < 34)) | ((h[row, col] > 100) & (h[row, col] < 124))) & (
                s[row, col] > 70) & (v[row, col] > 70):
            cv2.floodFill(src_image, mask, (col, row), (255, 255, 255), (loDiff,) * 3, (upDiff,) * 3, flags)
            cv2.circle(flood_img, center=(col, row), radius=2, color=(0, 0, 255), thickness=2)
            seed_cnt += 1
            if seed_cnt >= valid_seed_num:
                break
    # adjusting #
    show_seed = np.random.uniform(1, 100, 1).astype(np.uint16)
    cv2.namedWindow("floodfill" + str(show_seed), 0)
    cv2.resizeWindow("floodfill" + str(show_seed), 640, 480)
    cv2.imshow('floodfill' + str(show_seed), flood_img)
    cv2.namedWindow("flood_mask" + str(show_seed), 0)
    cv2.resizeWindow("flood_mask" + str(show_seed), 640, 480)
    cv2.imshow('flood_mask' + str(show_seed), mask)
    # adjusting #
    mask_points = []
    for row in range(1, img_h + 1):
        for col in range(1, img_w + 1):
            if mask[row, col] != 0:
                mask_points.append((col - 1, row - 1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))  # get the mask area
    if verify_scale(mask_rotateRect):
        return True, mask_rotateRect
    else:
        return False, mask_rotateRect


def locate_carPlate(orig_img, pred_image):
    carPlate_list = []
    temp1_orig_img = orig_img.copy()  # adjusting
    temp2_orig_img = orig_img.copy()  # adjusting
    # cloneImg, contours, heriachy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    for i, contour in enumerate(contours):  # filter the possible rectangle contours
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        rotate_rect = cv2.minAreaRect(contour)  # find the min area rectangle
        if verify_scale(rotate_rect):  # verify car plate by scale
            ret, rotate_rect2 = verify_color(rotate_rect, temp2_orig_img)  # Verify car plate by color
            if not ret:
                continue
            car_plate = img_Transform(rotate_rect2, temp2_orig_img)  # transform image if there is an angel
            car_plate = cv2.resize(car_plate, (car_plate_w, car_plate_h))  # adjust image size for CNN recognition
            # adjust #
            box = cv2.boxPoints(rotate_rect2).astype(int)
            for k in range(4):
                n1, n2 = k % 4, (k + 1) % 4
                # print(box[n1][0], box[n1][1], box[n2][0], box[n2][1])
                cv2.line(temp1_orig_img, (box[n1][0], box[n1][1]), (box[n2][0], box[n2][1]), (255, 0, 0), 2)
            cv2.imshow('opencv' + str(i), car_plate)
            # adjust #
            carPlate_list.append(car_plate)
    cv2.namedWindow("contour", 0)
    cv2.resizeWindow("contour", 640, 480)
    cv2.imshow('contour', temp1_orig_img)

    return carPlate_list


if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    car_plate_w, car_plate_h = 136, 36
    char_w, char_h = 20, 20
    img = cv2.imread('../images/pictures/48.jpg')

    pred_img = pre_process(img)  # preprocessing

    car_plate_list = locate_carPlate(img, pred_img)  # locating the car plate

    cv2.waitKey(0)
