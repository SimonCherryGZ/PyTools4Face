def get_obj_indices_size(path):
    size = 0
    with open(path, 'r') as f:
        data = f.readlines()

        for line in data:
            odom = line.split()
            prefix = odom[0]
            if prefix.startswith('v', 0, len(prefix)) \
                    and not prefix.startswith('vt', 0, len(prefix)) \
                    and not prefix.startswith('vn', 0, len(prefix)):
                size += 1

    print size
    return size


def read_obj_indices(path):
    tmp = []
    with open(path, 'r') as f:
        data = f.readlines()

        for line in data:
            odom = line.split()
            prefix = odom[0]
            if prefix.startswith('f', 0, len(prefix)):
                odom.remove('f')
                # print odom
                # tmp.append(odom)
                for element in odom:
                    pos = element.index('/')
                    # print pos
                    element = element[0:pos]
                    # print element
                    tmp.append(element)

    return tmp


def order_obj_indices(indices, size):
    lists = [[] for t in range(size)]

    i = 1
    for index in indices:
        print index
        lists[int(index) - 1].append(i)
        i += 1

    # print lists
    for element in lists:
        print element


if __name__ == '__main__':
    obj_dir = 'input/obj/'
    obj_name = 'base_face_uv.txt'
    obj_path = obj_dir + obj_name
    obj_indices = read_obj_indices(obj_path)
    indices_size = get_obj_indices_size(obj_path)

    order_obj_indices(obj_indices, indices_size)
