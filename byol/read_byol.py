import torch
import torchvision
import pickle

weight_path = 'pretrain_res50x1.pkl'






if __name__ == '__main__':
    with open(weight_path, 'rb') as f:
        data = pickle.load(f)
        print(data)
