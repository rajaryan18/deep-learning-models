from pathlib import Path
import os, random, shutil

def combine_dirs(path):
    cnt = 0
    imgs = list(Path(path).glob('*/*.png'))
    for img in imgs:
        shutil.move(img, f'{path}{cnt}.png')
        cnt += 1

def seperate(path, imgs, split):
    os.mkdir(path=path)
    for i in range(split):
        img = random.choice(imgs)
        shutil.move(img, f'{path}{i}.png')
        imgs.remove(img)
    return imgs

def create_seperation():
    try:
        os.mkdir(path='./data/test/')
        os.mkdir(path='./data/val/')
    except:
        pass
    imgs = list(Path('./data/train/vehicles/').glob('*.png'))
    split = int(len(imgs)*0.2)
    imgs = seperate('./data/test/vehicles/', imgs, split)
    imgs = seperate('./data/val/vehicles/', imgs, split)

    imgs = list(Path('./data/train/non-vehicles/').glob('*.png'))
    split = int(len(imgs)*0.2)
    imgs = seperate('./data/test/non-vehicles/', imgs, split)
    imgs = seperate('./data/val/non-vehicles/', imgs, split)