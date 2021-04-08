import numpy as np
import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

from os.path import join

classes = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
           'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
           'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
           'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop',
           'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer',
           'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
           'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']
if __name__ == '__main__':
    save_root = '/home/vision/jhkim/results/dsbn_ori/jigsaw_result/jigsaw_ssl_r_c/stage2/stage3_c_mat.npy'
    print(os.path.isdir(save_root))
    c_mat = np.load(save_root)
    print(c_mat)
    confusion_mat = pd.DataFrame(c_mat, index=classes, columns=classes)
    plt.figure(figsize = (10, 7))
    sn.heatmap(confusion_mat, annot=False)

    f_name = save_root.replace('npy','png')
    plt.savefig(f_name)
    plt.show()