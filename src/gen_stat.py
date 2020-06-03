"""
Image Similarity using Deep Ranking.

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

import numpy as np
from PIL import Image
import os


def gen_mean_std():
    """Generate mean and std for the image sizes of the WFood dataset."""
    image_list = []
    with open("./train_triplets.txt") as f:
        lines = [line.rstrip('\n').split(" ") for line in f]
        for line in lines:
            image_list.append(line[0])
            image_list.append(line[1])
            image_list.append(line[2])

    image_shapes = []

    cnt = 1
    for image in image_list:
        print(cnt)
        cnt+=1
        img_name = os.path.join('../food', image) + '.jpg'
        img = Image.open(img_name)
        image_shapes.append([img.width, img.height])

    mean = np.mean(image_shapes, axis=0)
    std = np.std(image_shapes, axis=0)

    return mean, std


if __name__ == '__main__':
    mean, std = gen_mean_std()
    print(mean, std)
    # mean : 457, 308
    # std  : 25, 15
