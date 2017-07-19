
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


def order_obj_indices(indices):
    for index in indices:
        print index


if __name__ == '__main__':

    obj_dir = 'input/obj/'
    obj_name = 'base_face_uv.txt'
    obj_path = obj_dir + obj_name
    obj_indices = read_obj_indices(obj_path)
    order_obj_indices(obj_indices)
