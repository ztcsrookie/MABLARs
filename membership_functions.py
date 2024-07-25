def triangle_mf(x, a, b, c):
    '''
    :param x: the input value
    :param a: the left value of the triangle membership function
    :param b: the middle value of the triangle membership function
    :param c: the right value of the triangle membership function
    :return: md_x: the membership degree of x to the triangle membership function
    '''
    if x <= a:
        md_x = 0
    elif a < x <= b:
        md_x = (x - a) / (b - a)
    elif b < x <= c:
        md_x = (c - x) / (c - b)
    else:
        md_x = 0
    return md_x


def left_shoulder_mf(x,a,b):
    '''
    下降那个
    :param x: the input value
    :param a: the left value of the left shoulder membership function
    :param b: the right value of the left shoulder membership function
    :return: md_x: the membership degree of x to the left shoulder membership function
    '''
    if x <= a:
        md_x = 1
    elif a < x < b:
        md_x = (b - x) / (b - a)
    else:
        md_x = 0
    return md_x

def right_shoulder_mf(x,a,b):
    '''
    上升那个
    :param x: the input value
    :param a: the left value of the right shoulder membership function
    :param b: the right value of the right shoulder membership function
    :return: md_x: the membership degree of x to the right shoulder membership function
    '''
    if x <= a:
        md_x = 0
    elif a < x < b:
        md_x = (x - a) / (b - a)
    else:
        md_x = 1
    return md_x