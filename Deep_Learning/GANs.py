
from PIL import Image
from tqdm import tqdm
import glob
import numpy as np
import torch.nn as nn
import torch
from torchvision.models import vgg19
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import argparse

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)

        return x

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
padsize = 150

class testImageDataset(Dataset):
    def __init__(self, root):
        
        self.tensor_setup = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        filePath = self.files[index % len(self.files)]
        R = np.array(Image.open(filePath),'f') / 255.
        R = np.pad(R,[(padsize,padsize),(padsize,padsize),(0,0)],'symmetric')

        return {"R": self.tensor_setup(R[:,:,:3]), "Name": os.path.basename(filePath).split(".")[0]}

    def __len__(self):
        return len(self.files)

class GCVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.LeakyReLU(negative_slope=0.01,inplace=True)):
        super(GCVGGBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            act_func,
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            act_func
        )

    def forward(self, x):
        out = self.model(x)

        return out

class GCNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GCNet, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = Interpolate(scale_factor=2, mode='bilinear')

        self.conv0_0 = GCVGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = GCVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = GCVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = GCVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = GCVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = GCVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = GCVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = GCVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = GCVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = GCVGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = GCVGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = GCVGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = GCVGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = GCVGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = GCVGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final1 = nn.Sequential(
            nn.Conv2d(nb_filter[0], out_channels, kernel_size=3, padding=1),
            )
        self.final2 = nn.Sequential(
            nn.Conv2d(nb_filter[0], out_channels, kernel_size=3, padding=1),
            )
        self.final3 = nn.Sequential(
            nn.Conv2d(nb_filter[0], out_channels, kernel_size=3, padding=1),
            )
        self.final4 = nn.Sequential(
            nn.Conv2d(nb_filter[0], nb_filter[0], 5, padding=2),
            nn.BatchNorm2d(nb_filter[0]),
            nn.LeakyReLU(negative_slope=0.01,inplace=True),
            nn.Conv2d(nb_filter[0], out_channels, kernel_size=3, padding=1),
            )


        self.G_x_D = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_y_D = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_x_G = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)
        self.G_y_G = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=0,bias=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output4 = self.final4(x0_4)

        return output4

def convert_to_numpy(input,H,W):
    image = input[:,:,padsize:H-padsize,padsize:W-padsize].clone()
    input_numpy = image[:,:,:H,:W].clone().cpu().numpy().reshape(3,H-padsize*2,W-padsize*2).transpose(1,2,0)
    for i in range(3):
        input_numpy[:,:,i] = input_numpy[:,:,i] * std[i] + mean[i]
    return  input_numpy

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--path", required = True, help = "path to input dataset")
opt = parser.parse_args()
dataset_name = opt.path
os.makedirs(dataset_name + "/output", exist_ok = True)


print("INFO Creating Model")
Generator = GCNet()
print("INFO Loading weights")
Generator.load_state_dict(torch.load("weight.pth", map_location = 'cpu'))

print("INFO Loading dataset")
image_dataset = testImageDataset(dataset_name)
print("[Dataset name: %s] --> %d images" % (dataset_name, len(image_dataset)))

for image_num in tqdm(range(len(image_dataset))):
    data = image_dataset[image_num]
    R = data["R"]
    _,first_h,first_w = R.size()
    R = torch.nn.functional.pad(R,(0,(R.size(2)//16)*16+16-R.size(2),0,(R.size(1)//16)*16+16-R.size(1)),"constant")
    R = R.view(1,3,R.size(1),R.size(2))
    
    print("INFO Removing Reflection from Image")
    with torch.no_grad():
        output  = Generator(R) 
    output_np = np.clip(convert_to_numpy(output,first_h,first_w) + 0.015,0,1)
    R_np = convert_to_numpy(R,first_h,first_w)
    final_output = np.fmin(output_np, R_np)
    
    print("INFO Saving Image")
    Image.fromarray(np.uint8(final_output * 255)).save(dataset_name + "/output/" + data["Name"] + ".png")
