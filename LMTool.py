import cv2
import numpy as np


def read_points(path):
    landmarks = []
    with open(path) as txt_file:
        for line in txt_file:
            x, y = line.split()
            landmarks.append((int(x), int(y)))

    return landmarks


def read_z_aix(path):
    tmp = []
    with open(path, 'r') as f:
        data = f.readlines()

        for line in data:
            odom = line.split()
            numbers_float = map(float, odom)
            # print numbers_float[0]
            tmp.append(float(numbers_float[0]))

    return tmp


def read_image(path):
    img = cv2.imread(path)
    img = np.float32(img) / 255.0
    return img


def get_map_index(x):
    return {
        0: 27,
        1: 28,
        4: 29,
        6: 30,
        8: 19,
        10: 25,
        12: 24,
        15: 23,
        16: 22,
        18: 26,
        20: 4,
        23: 6,
        25: 21,
        27: 5,
        31: 13,
        33: 14,
        35: 8,
        36: 0,
        39: 1,
        42: 7,
        45: 10,
        48: 16,
        54: 17,
        57: 18,
        62: 15,
        -1: -1
    }.get(x, -1)


if __name__ == '__main__':

    # name = '20130821040137899'
    name = 'obama2'
    image_path = 'output/texture/' + name + '.png'
    points_path = 'output/texture/' + name + '.txt'
    z_path = 'input/texture/z_axis.txt'
    tmp_path = 'output/texture/' + name + '_t.txt'
    output_path = 'output/texture/' + name + '_e.txt'

    points = read_points(points_path)
    z_axis = read_z_aix(z_path)
    vertices = []

    tmp = [0] * 40
    for index in range(len(tmp)):
        tmp[index] = (0, 0)

    for index in range(len(points)):
        print "index=%s" % index
        map_index = get_map_index(index)
        print "map_index=%s" % map_index
        if map_index >= 0:
            tmp[map_index] = points[index]

    for index in range(len(tmp)):
        if index == 20:
            x1, y1 = points[20]
            x2, y2 = points[23]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 12:
            x1, y1 = points[31]
            x2, y2 = points[35]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 3:
            x1, y1 = points[37]
            x2, y2 = points[38]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 2:
            x1, y1 = points[40]
            x2, y2 = points[41]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 9:
            x1, y1 = points[43]
            x2, y2 = points[44]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 11:
            x1, y1 = points[46]
            x2, y2 = points[47]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 38:
            x1, y1 = points[36]
            x2, y2 = points[39]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 39:
            x1, y1 = points[42]
            x2, y2 = points[45]
            tmp[index] = ((x1+x2)/2, (y1+y2)/2)
        elif index == 37:
            tmp[index] = tmp[28]
        elif index == 35:
            tmp[index] = tmp[29]
        elif index == 33:
            tmp[index] = tmp[30]
        elif index == 31:
            tmp[index] = tmp[19]
        elif index == 32:
            tmp[index] = tmp[25]
        elif index == 34:
            tmp[index] = tmp[24]
        elif index == 36:
            tmp[index] = tmp[23]

    np.savetxt(tmp_path, tmp, fmt="%d")

    (chin_x, chin_y) = points[8]

    (x0, y0) = tmp[0]
    scale = (((x0 - chin_x) / float(30)) / -2.15)

    for index in range(len(tmp)):
        z = z_axis[index] * scale
        (x, y) = tmp[index]
        x = (x - chin_x) / float(30)
        y = (y - chin_y) / float(-30)
        print "x=%s, y=%s, z=%s" % (x, y, z)
        vertices.append((float(x), float(y), float(z)))

    # for (x, y, z) in vertices:
    #     print "x=%s, y=%s, z=%s" % (x, y, z)
    np.savetxt(output_path, vertices, fmt="v %.6f %.6f %.6f")
