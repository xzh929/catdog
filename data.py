from torch.utils.data import Dataset,DataLoader
import os
import cv2
import numpy as np
import torch


class dataset(Dataset):
    def __init__(self,root,is_train=True):
        self.img_data = []
        sub_dir = "img" if is_train else "img2"
        for filename in os.listdir(f"{root}/{sub_dir}"):
            file_list = filename.split(".")
            tag = file_list[0]
            img_path = f"{root}/{sub_dir}/{filename}"
            self.img_data.append((img_path,tag))

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, item):
        data = self.img_data[item]
        tag = data[1]
        img = cv2.imread(data[0])
        img = img.transpose(2,1,0)
        img = img/255
        tag = float(tag)
        tag = torch.tensor(tag)
        tag = tag.to(torch.int64) # tag转tensor.int类型
        return np.float32(img),tag


if __name__ == '__main__':
    train_data = dataset("D:\data\cat_dog",is_train=True)
    print(train_data[0][1])

