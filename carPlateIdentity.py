import cv2
import os
import sys
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']


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

    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1 / r)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if min_area < area < max_area and min_aspect < r < max_aspect:
        # check if the angel of rectangle exceeds theta
        # if ((rotate_rect[1][0] < rotate_rect[1][1] and -90 <= rotate_rect[2] < -(90 - theta)) or
        #         (rotate_rect[1][1] < rotate_rect[1][0] and -theta < rotate_rect[2] <= 0)):
        if ((rotate_rect[1][0] < rotate_rect[1][1] and -90 <= -rotate_rect[2] < -(90 - theta)) or
                (rotate_rect[1][1] < rotate_rect[1][0] and -theta < -rotate_rect[2] <= 0)):  # add minus to
            return True
    return False


global car_img


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
        car_img = image[int(car_rect[0][1] - rect_h / 2): int(car_rect[0][1] + rect_h / 2),
                        int(car_rect[0][0] - rect_w / 2): int(car_rect[0][0] + rect_w / 2)]
        return car_img

    car_rect = (car_rect[0], (rect_w, rect_h), angle)
    box = cv2.boxPoints(car_rect)

    height_point = right_point = [0, 0]
    left_point = low_point = [car_rect[0][0], car_rect[0][1]]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if height_point[1] < point[1]:
            height_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # positive angel
        new_right_point = [right_point[0], height_point[1]]
        pts1 = np.float32([left_point, height_point, right_point])
        pts2 = np.float32([left_point, height_point, new_right_point])
        M = cv2.getAffineTransform(pts1, pts2)  # get affine transform matrix
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))  # apply transformation
        car_img = dst[int(left_point[1]):int(height_point[1]), int(left_point[0]):int(new_right_point[0])]

    elif left_point[1] > right_point[1]:  # negative angel
        new_left_point = [left_point[0], height_point[1]]
        pts1 = np.float32([left_point, height_point, right_point])
        pts2 = np.float32([new_left_point, height_point, right_point])
        M = cv2.getAffineTransform(pts1, pts2)  # get affine transform matrix
        dst = cv2.warpAffine(image, M, (round(img_w * 2), round(img_h * 2)))  # apply transformation
        car_img = dst[int(right_point[1]):int(height_point[1]), int(new_left_point[0]):int(right_point[0])]

    return car_img


def pre_process(orig_img):
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)  # convert to grayscale image
    # cv2.imshow('gray_img', gray_img)

    # blur_img = cv2.blur(gray_img, (3, 3))
    kernel_size = 5
    gauss_gray = gaussian_blur(gray_img, kernel_size)  # gaussian blue
    # cv2.imshow('blur', blur_img)

    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray, low_threshold, high_threshold)
    # cv2.imshow('canny', canny_edges)

    # sobel_img = cv2.Sobel(blur_img, cv2.CV_16S, 1, 0, ksize=3)
    # sobel_img = cv2.convertScaleAbs(sobel_img)  # sobel edge finding
    # cv2.imshow('sobel', sobel_img)

    hsv_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)  # convert to HSV image
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    blue_img = (((h > 26) & (h < 34)) | ((h > 100) & (h < 124))) & (s > 70) & (v > 70)
    blue_img = blue_img.astype('float32')  # find blue area [100, 124]

    mix_img = np.multiply(canny_edges, blue_img)  # find blue edge
    # cv2.imshow('mix', mix_img)

    mix_img = mix_img.astype(np.uint8)

    ret1, binary_img = cv2.threshold(mix_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # binarization
    # cv2.imshow('binary',binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)  # morphology operation
    # cv2.imshow('close', close_img)

    return close_img


# flood filling the image and remove the image which is not a car plate
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
        # row, col = points_row[rand_index], points_col[rand_index]
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
    # show_seed = np.random.uniform(1, 100, 1).astype(np.uint16)
    # cv2.namedWindow("floodfill" + str(show_seed), 0)
    # cv2.resizeWindow("floodfill" + str(show_seed), 640, 480)
    # cv2.imshow('floodfill' + str(show_seed), flood_img)
    # cv2.namedWindow("flood_mask" + str(show_seed), 0)
    # cv2.resizeWindow("flood_mask" + str(show_seed), 640, 480)
    # cv2.imshow('flood_mask' + str(show_seed), mask)
    # adjusting #
    # find the minimum rectangle area
    mask_points = []
    for row in range(1, img_h + 1):
        for col in range(1, img_w + 1):
            if mask[row, col] != 0:
                mask_points.append((col - 1, row - 1))
    mask_rotateRect = cv2.minAreaRect(np.array(mask_points))
    if verify_scale(mask_rotateRect):
        return True, mask_rotateRect
    else:
        return False, mask_rotateRect


# locating car plate
def locate_carPlate(orig_img, pred_image):
    carPlate_list = []
    temp1_orig_img = orig_img.copy()  # adjusting
    temp2_orig_img = orig_img.copy()  # adjusting
    contours, hierarchy = cv2.findContours(pred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    for i, contour in enumerate(contours):
        cv2.drawContours(temp1_orig_img, contours, i, (0, 255, 255), 2)
        rotate_rect = cv2.minAreaRect(contour)  # find the min area rectangle
        if verify_scale(rotate_rect):  # verify car plate by scale
            ret1, rotate_rect2 = verify_color(rotate_rect, temp2_orig_img)  # Verify car plate by color
            if not ret1:
                continue
            car_plate1 = img_Transform(rotate_rect2, temp2_orig_img)  # transform image if there is an angel
            car_plate1 = cv2.resize(car_plate1, (car_plate_w, car_plate_h))  # adjust image size for CNN recognition
            # adjust area #
            # box = cv2.boxPoints(rotate_rect2)
            box = cv2.boxPoints(rotate_rect2).astype(int)
            for k in range(4):
                n1, n2 = k % 4, (k + 1) % 4
                cv2.line(temp1_orig_img, (box[n1][0], box[n1][1]), (box[n2][0], box[n2][1]), (255, 0, 0), 2)
            # cv2.imshow('opencv_' + str(i), car_plate1)
            # adjust area #
            carPlate_list.append(car_plate1)
    # cv2.namedWindow("contour", 0)
    # cv2.resizeWindow("contour", 640, 480)
    # cv2.imshow('contour', temp1_orig_img)
    return carPlate_list


# cut the edge of the character
def horizontal_cut_chars(plate):
    char_addr_list = []
    area_left, area_right, char_left, char_right = 0, 0, 0, 0
    img_w = plate.shape[1]

    def getColSum(img1, col1):
        sum1 = 0
        for n in range(img1.shape[0]):
            sum1 += round(img1[n, col1] / 255)
        return sum1

    sum2 = 0
    for col in range(img_w):
        sum2 += getColSum(plate, col)
    col_limit = 0  # round(0.5*sum/img_w)
    charWid_limit = [round(img_w / 12), round(img_w / 5)]
    is_char_flag = False

    for i in range(img_w):
        colValue = getColSum(plate, i)
        if colValue > col_limit:
            if not is_char_flag:
                area_right = round((i + char_right) / 2)
                area_width = area_right - area_left
                char_width = char_right - char_left
                if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
                    char_addr_list.append((area_left, area_right, char_width))
                char_left = i
                area_left = round((char_left + char_right) / 2)
                is_char_flag = True
        else:
            if is_char_flag:
                char_right = i - 1
                is_char_flag = False

    if area_right < char_left:
        area_right, char_right = img_w, img_w
        area_width = area_right - area_left
        char_width = char_right - char_left
        if (area_width > charWid_limit[0]) and (area_width < charWid_limit[1]):
            char_addr_list.append((area_left, area_right, char_width))
    return char_addr_list


def get_chars(car_plate1):
    img_h, img_w = car_plate1.shape[:2]
    h_proj_list = []  # horizontal projection
    h_temp_len, v_temp_len = 0, 0
    h_startIndex, h_end_index = 0, 0  # horizontal index
    h_proj_limit = [0.2, 0.8]  # filter the one which projection exceed the limit
    char_imgs = []

    # project to Y and get the continues length, could be more than one
    h_count = [0 for _ in range(img_h)]
    for row in range(img_h):
        temp_cnt = 0
        for col in range(img_w):
            if car_plate1[row, col] == 255:
                temp_cnt += 1
        h_count[row] = temp_cnt
        if temp_cnt / img_w < h_proj_limit[0] or temp_cnt / img_w > h_proj_limit[1]:
            if h_temp_len != 0:
                h_end_index = row - 1
                h_proj_list.append((h_startIndex, h_end_index))
                h_temp_len = 0
            continue
        if temp_cnt > 0:
            if h_temp_len == 0:
                h_startIndex = row
                h_temp_len = 1
            else:
                h_temp_len += 1
        else:
            if h_temp_len > 0:
                h_end_index = row - 1
                h_proj_list.append((h_startIndex, h_end_index))
                h_temp_len = 0

    # get the total length
    if h_temp_len != 0:
        h_end_index = img_h - 1
        h_proj_list.append((h_startIndex, h_end_index))

    # get the biggest length projection
    h_maxIndex, h_maxHeight = 0, 0
    for i, (start, end) in enumerate(h_proj_list):
        if h_maxHeight < (end - start):
            h_maxHeight = (end - start)
            h_maxIndex = i
    if h_maxHeight / img_h < 0.5:
        return char_imgs
    chars_top, chars_bottom = h_proj_list[h_maxIndex][0], h_proj_list[h_maxIndex][1]

    plates = car_plate1[chars_top:chars_bottom + 1, :]
    # cv2.imshow('plates',plates)
    cv2.imwrite('./carIdentityData/opencv_output/car.jpg', car_plate1)
    cv2.imwrite('./carIdentityData/opencv_output/plate.jpg', plates)
    char_addr_list = horizontal_cut_chars(plates)  # cut into single character

    for i, addr in enumerate(char_addr_list):
        char_img = car_plate1[chars_top:chars_bottom + 1, addr[0]:addr[1]]
        char_img = cv2.resize(char_img, (char_w, char_h))
        char_imgs.append(char_img)
        # cv2.imshow("char_img", char_img)
        # cv2.waitKey()
    return char_imgs


def extract_char(car_plate1):
    gray_plate = cv2.cvtColor(car_plate1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_plate', gray_plate)
    ret1, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('binary_plate', binary_plate)
    char_img_list1 = get_chars(binary_plate)  # recognize character
    return char_img_list1


def cnn_select_carPlate(plate_list, model_path):
    if len(plate_list) == 0:
        return False, plate_list
    g1 = tf.Graph()
    sess1 = tf.Session(graph=g1)
    with sess1.as_default():
        with sess1.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess1, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net1_x_place = graph.get_tensor_by_name('x_place:0')
            net1_keep_place = graph.get_tensor_by_name('keep_place:0')
            net1_out = graph.get_tensor_by_name('out_put:0')

            input_x = np.array(plate_list)
            net_outs = tf.nn.softmax(net1_out)
            preds = tf.argmax(net_outs, 1)  # predict
            probs = tf.reduce_max(net_outs, reduction_indices=[1])  # get the probability
            pred_list, prob_list = sess1.run([preds, probs], feed_dict={net1_x_place: input_x, net1_keep_place: 1.0})
            # find the plate with the biggest probability
            result_index, result_prob = -1, 0.
            for i, pred in enumerate(pred_list):
                if pred == 1 and prob_list[i] > result_prob:
                    result_index, result_prob = i, prob_list[i]
            if result_index == -1:
                return False, plate_list[0]
            else:
                return True, plate_list[result_index]


def cnn_recognize_char(img_list, model_path):
    g2 = tf.Graph()
    sess2 = tf.Session(graph=g2)
    text_list = []

    if len(img_list) == 0:
        return text_list
    with sess2.as_default():
        with sess2.graph.as_default():
            model_dir = os.path.dirname(model_path)
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess2, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            net2_x_place = graph.get_tensor_by_name('x_place:0')
            net2_keep_place = graph.get_tensor_by_name('keep_place:0')
            net2_out = graph.get_tensor_by_name('out_put:0')

            data = np.array(img_list)
            # recognize character
            net_out = tf.nn.softmax(net2_out)
            preds = tf.argmax(net_out, 1)
            my_preds = sess2.run(preds, feed_dict={net2_x_place: data, net2_keep_place: 1.0})

            for i in my_preds:
                text_list.append(char_table[i])
            return text_list


if __name__ == '__main__':
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    car_plate_w, car_plate_h = 136, 36
    char_w, char_h = 20, 20
    plate_model_path = './carIdentityData/model/plate_recongnize/model.ckpt-510.meta'
    char_model_path = './carIdentityData/model/char_recongnize/model.ckpt-600.meta'
    img = cv2.imread('../images/pictures/1.jpg')

    pred_img = pre_process(img)  # preprocessing

    car_plate_list = locate_carPlate(img, pred_img)  # locating the car plate

    ret, car_plate = cnn_select_carPlate(car_plate_list, plate_model_path)  # car plate recognize
    if not ret:
        print("car plate not found!")
        sys.exit(-1)
    # cv2.imshow('cnn_plate', car_plate)

    char_img_list = extract_char(car_plate)  # extract character

    text = cnn_recognize_char(char_img_list, char_model_path)  # recognize car plate character
    print(text)

    cv2.waitKey(0)
