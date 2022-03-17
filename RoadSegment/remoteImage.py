from glob import glob
from PIL import Image
import os

def get_files(data_folder):
        #
        return  glob("{}/*.{}".format(data_folder, 'jpg'))

def get_label_file(data_path, data_dir, label_dir):
    #
    data_path = data_path.replace(data_dir, label_dir)
    frame, ext = data_path.split('.')
    return "{}.{}".format(frame, 'png')

DATA_PATH = os.path.join(os.getcwd(), 'dataset\\')
train_path, val_path, test_path = [os.path.join(DATA_PATH, x) for x in ['image','val','test']]

data_files = get_files(train_path)
label_files = [get_label_file(f, 'image', 'labeltrain') for f in data_files]

for i, img in enumerate(label_files):
    try:
        label = Image.open(img)
    except:
        os.remove(data_files[i])
    