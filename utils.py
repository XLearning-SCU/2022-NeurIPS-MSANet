import os, sys
import random, logging
import torchvision.transforms as transforms

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def logger(log_file=None):
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, 
        format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p'
    )
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logging.getLogger().addHandler(fh)
    return logging

def RandomRot(img, angle=90, p=0.5):
    if random.random() > p:
        return transforms.functional.rotate(img, angle)
    return img