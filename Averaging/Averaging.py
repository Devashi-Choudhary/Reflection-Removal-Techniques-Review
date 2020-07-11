import argparse
import cv2
import numpy as np
import os
import glob

def Averaging(files):
    imgs=[]
    for f in files:
        imgs.append(np.array(cv2.imread(f).astype('uint16')))
    res=np.zeros(imgs[0].shape)
    res=imgs[0]+imgs[1]
    i=2
    print(len(imgs))
    while(i<len(imgs)):
        res=res+imgs[i]
        i+=1
    res=res/len(imgs)
    return res

def read(imagepath):
    files = [f for f in glob.glob(imagepath + "/*")]
    res1 = Averaging(files)
    cv2.imshow("output", res1)
    cv2.waitKey(0)
    cv2.imwrite('Average.png', res1)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagepath", required = True, help = "path to input image directory")
args = ap.parse_args()

if __name__ == "__main__":
    imagepath = args.imagepath
    read(imagepath)