import numpy as np
import torch
from PIL import Image

def load_obj_data_list(list_pth):
    """load img from pth list
    :return: list[img_path,label]"""
    content = np.loadtxt(list_pth,dtype=str)
    img_pths = content[:,0]
    labels = content[:,1]
    return list(img_pths),list(labels)

def load_img(img_pth,):
    """load img"""
    if img_pth[-1] != "g":
        return Image.open(img_pth+"g")
    return Image.open(img_pth)

def one_hot(K,label):
    return torch.eye(K,dtype=torch.int8)[torch.tensor(label)]


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s
