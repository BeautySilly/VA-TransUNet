import h5py
import numpy as np
import glob
import scipy.misc as smc
from   imageio import imsave

test_dir = '../data/Synapse/test_vol_h5/'
save_dir = './visualize/test/'

files = glob.glob(test_dir + '*.h5')
files.sort()
print(files)

for file in files:
    file_name = file.split('/')[-1].split('.')[0]
    print(file_name)
    data = h5py.File(file, 'r')
    image = data['image']
    label = data['label']
    for i in range(image.shape[0]):
        print(image[i, ...].shape)
        image_arr = image[i, ...]
        label_arr = label[i, ...]
        label_arr[label_arr != 0] = 1
        image_name = "{0}_slice{1:0>3}_image.png".format(file_name, i)
        label_name = "{0}_slice{1:0>3}_label.png".format(file_name, i)
        imsave(save_dir + 'image/' + image_name, image_arr)
        imsave(save_dir + 'label/' + label_name, label_arr)



