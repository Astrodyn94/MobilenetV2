import torch.utils.data as data
import torchvision.transforms as transforms
from skimage.transform import rescale
import numpy as np
import scipy.misc
import os
import torch 

def save_model(model, epoch , opt):
    model_out_path = opt.name + '.pth'
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images


class caltech_dataloader(data.Dataset):
    def __init__(self, opt , phase = 'train'):
        root = opt.dataroot + phase
        _, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        
        if len(imgs)==0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
        
        self.imgs = imgs

    def __getitem__(self, index):
        
        path, label = self.imgs[index]
        name = os.path.basename(path)
        img_original = scipy.misc.imread(path)
        return label, name, np.transpose(img_original ,(2,0,1))
               
    def __len__(self):
        return len(self.imgs)

