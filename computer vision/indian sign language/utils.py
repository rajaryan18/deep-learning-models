import random, os
from pathlib import Path
import shutil

def generate_test_dataset():
    image_path = Path("./Indian/")
    for dirs in os.walk("./Indian/"):
        subdir = dirs[0].split('/')[2]
        if subdir is "":
            continue
        try:
            os.mkdir(path=f"./data/test/{subdir}/")
            os.mkdir(path=f"./data/val/{subdir}/")
        except:
            pass
        image_path_list = list(Path(dirs[0]).glob("*.jpg"))
        print(f"Length of {subdir} is {len(image_path_list)}")
        for i in range(0, 200):
            random_img = random.choice(image_path_list)
            shutil.move(random_img, f"./data/val/{subdir}/{i}.jpg")
            image_path_list.remove(random_img)
        for i in range(0, 200):
            random_img = random.choice(image_path_list)
            shutil.move(random_img, f"./data/test/{subdir}/{i}.jpg")
            image_path_list.remove(random_img)