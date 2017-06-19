#!/usr/bin/env python
# coding=utf-8

import os
import sys
import cv2
import dlib
import numpy as np
import math
from skimage import io
import time


predictor_path = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_landmarks(im):
    rects = detector(im, 1)
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def read_points(path):
    landmarks = []
    with open(path) as txt_file:
        for line in txt_file:
            x, y = line.split()
            landmarks.append((int(x), int(y)))
    return landmarks


def read_image(path):
    img = cv2.imread(path)
    img = np.float32(img) / 255.0
    return img


def remap_uv_vertices(vertices, index):
    print index
    if index == 0:
        return vertices[18]
    elif index == 1:
        return vertices[0]
    elif index == 2:
        return vertices[36]
    elif index == 3:
        (x0, y0) = vertices[0]
        (x1, y1) = vertices[1]
        (x17, y17) = vertices[17]
        return x17, (y0 + y1) / 2
    elif index == 4:
        (x3, y3) = vertices[3]
        (x5, y5) = vertices[5]
        return x5, y3
    elif index == 5:
        (x5, y5) = vertices[5]
        (x6, y6) = vertices[6]
        return x6, y5
    elif index == 6:
        return vertices[48]
    elif index == 7:
        (x5, y5) = vertices[5]
        (x6, y6) = vertices[6]
        (x8, y8) = vertices[8]
        return x8, (y5 + y6) / 2
    elif index == 8:
        return vertices[57]
    elif index == 9:
        return vertices[62]
    elif index == 10:
        return vertices[33]
    elif index == 11:
        return vertices[31]
    elif index == 12:
        (x31, y31) = vertices[31]
        (x35, y35) = vertices[35]
        return (x31 + x35) / 2, (y31 + y35) / 2
    elif index == 13:
        return vertices[27]
    elif index == 14:
        (x40, y40) = vertices[40]
        (x41, y41) = vertices[41]
        return (x40 + x41) / 2, (y40 + y41) / 2
    elif index == 15:
        return vertices[39]
    elif index == 16:
        return vertices[20]
    elif index == 17:
        (x37, y37) = vertices[37]
        (x38, y38) = vertices[38]
        return (x37 + x38) / 2, (y37 + y38) / 2
    elif index == 18:
        (x20, y20) = vertices[20]
        (x23, y23) = vertices[23]
        return (x20 + x23) / 2, (y20 + y23) / 2
    elif index == 19:
        return vertices[23]
    elif index == 20:
        return vertices[42]
    elif index == 21:
        return vertices[35]
    elif index == 22:
        (x46, y46) = vertices[46]
        (x47, y47) = vertices[47]
        return (x46 + x47) / 2, (y46 + y47) / 2
    elif index == 23:
        return vertices[45]
    elif index == 24:
        (x43, y43) = vertices[43]
        (x44, y44) = vertices[44]
        return (x43 + x44) / 2, (y43 + y44) / 2
    elif index == 25:
        return vertices[25]
    elif index == 26:
        return vertices[16]
    elif index == 27:
        (x26, y26) = vertices[26]
        (x15, y15) = vertices[15]
        (x16, y16) = vertices[16]
        return x26, (y15 + y16) / 2
    elif index == 28:
        return vertices[54]
    elif index == 29:
        (x10, y10) = vertices[10]
        (x12, y12) = vertices[12]
        return x10, y12
    elif index == 30:
        (x11, y11) = vertices[11]
        (x13, y13) = vertices[13]
        return x11, y13
    elif index == 31:
        return vertices[15]
    elif index == 32:
        return vertices[12]
    elif index == 33:
        return vertices[10]
    elif index == 34:
        return vertices[8]
    elif index == 35:
        return vertices[6]
    elif index == 36:
        return vertices[1]
    elif index == 37:
        return vertices[4]
    elif index == 38:
        (x36, y36) = vertices[36]
        (x39, y39) = vertices[39]
        return (x36 + x39) / 2, (y36 + y39) / 2
    elif index == 39:
        (x42, y42) = vertices[42]
        (x45, y45) = vertices[45]
        return (x42 + x45) / 2, (y42 + y45) / 2


if __name__ == '__main__':

    path = 'input/texture/'
    name = '20130821040137899'
    fmt = '.jpg'
    img_name = path + name + fmt
    text_name = path + name + '.txt'

    # 读取人脸图片
    image = read_image(img_name)
    # 创建1024 x 1024大小的空白图片
    output = np.zeros((1024, 1024, 3), np.float32())
    sp = image.shape
    img_w = sp[1]
    img_h = sp[0]
    roi_x = (1024 - img_w) / 2
    roi_y = (1024 - img_h) / 2
    # 将人脸图片置于空白图片中央，生成新图片
    output[roi_y:roi_y + img_h, roi_x:roi_x + img_w] = image
    output_path = 'output/texture/' + name + '_output.jpg'
    cv2.imwrite(output_path, output * 255)

    # 保存新图片的人脸关键点
    if not os.path.exists(text_name):
        img = io.imread(output_path)
        result = get_landmarks(img)
        np.savetxt(text_name, result, fmt="%d %d")

    # 根据人脸关键点，生成UV贴图坐标
    points = read_points(text_name)
    vertices = []
    for index in range(len(points)):
        (x, y) = points[index]
        x = float(x) / 1024
        y = 1 - (float(y) / 1024)
        vertices.append((float(x), float(y)))

    vertices_path = 'output/texture/' + name + '_vertices.txt'
    # np.savetxt(vertices_path, vertices, fmt="vt %.4f %.4f")

    # 贴图坐标重定向
    remap_uv = [0] * 40
    for i in range(0, 40):
        remap_uv[i] = remap_uv_vertices(vertices, i)

    vertices_remap_path = 'output/texture/' + name + '_vertices_remap.txt'
    np.savetxt(vertices_remap_path, remap_uv, fmt="vt %.4f %.4f")

    # 描绘关键点位置和序号
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.25
    fontcolor = (255, 0, 0)
    for index in range(len(points)):
        (x, y) = points[index]
        cv2.circle(output, (x, y), 2, (0, 255, 255), -1)
        cv2.putText(output, str(index), (x, y), fontface, fontscale, fontcolor)

    # 描绘人脸轮廓
    # pts = np.array([
    #     points[0], points[1], points[4], points[6],
    #     points[8], points[10], points[12], points[15],
    #     points[16], points[25], points[23], points[20],
    #     points[18]
    # ], np.int32)
    # pts = pts.reshape((-1, 1, 2))
    # image = cv2.polylines(image, [pts], True, (0, 255, 0))

    cv2.namedWindow("Image")
    cv2.imshow("Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
