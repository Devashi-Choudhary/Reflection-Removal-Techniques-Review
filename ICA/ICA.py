import argparse
import cv2
import numpy as np

def read(imagepath1, imagepath2):
    img1 = cv2.imread(imagepath1).astype(np.float32)
    img2 = cv2.imread(imagepath2).astype(np.float32)
    return img1, img2
    
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--imagepath1", required = True, help = "path to input image ")
ap.add_argument("-i2", "--imagepath2", required = True, help = "path to input image ")
args = ap.parse_args()

if __name__ == "__main__":
    imagepath1 = args.imagepath1
    imagepath2 = args.imagepath2
    img1, img2 = read(imagepath1, imagepath2)

img1=cv2.resize(img1,(1024,1024))
img2=cv2.resize(img2,(1024,1024))

def Estimate_Theta(r,phi,moment):
    x=(r**2)*np.sin(moment*phi)
    y=(r**2)*np.cos(moment*phi)
    a=x.sum()
    b=y.sum()
    Theta= (1/moment)*np.arctan2(a,b)
    return Theta    

def Estimate_Scaling(Theta,img1,img2):
    Sx = img1*np.cos(Theta)+img2*np.sin(Theta)
    Sy = img1*np.cos(Theta- np.pi / .2)+img2*np.sin(Theta- np.pi / .2)
    s1 = (Sx**2).sum()
    s2 = (Sy**2).sum()
    S = np.diag([1. / s1, 1. / s2]) 
    return S    

def decompose(Y1, Y2):
        Y1 -= Y1.mean()
        Y2 -= Y2.mean()

        R = Y1**2 + Y2**2
        PHI = np.arctan2(Y2, Y1)

        theta1=Estimate_Theta(R,PHI,2)
        
        R1inv = np.array([[np.cos(theta1), -np.sin(theta1)],[np.sin(theta1), np.cos(theta1)]]).transpose()                              

        Sinv =  Estimate_Scaling(theta1, Y1, Y2)                             

        theta2=Estimate_Theta(R,PHI,4)
        R2inv = np.array([[np.cos(theta2), -np.sin(theta2)],[np.sin(theta2), np.cos(theta2)]]).transpose()                                

        Minv = np.matmul(R2inv, np.matmul(Sinv, R1inv))                  

        return R1inv, Sinv, R2inv, Minv

R1inv, Sinv, R2inv, Minv = decompose(img1, img2)
Im = np.concatenate([img1.reshape(1, -1),img2.reshape(1, -1)], axis=0)
Im = np.matmul(Minv, Im)

i1 = Im[0, :]
i2 = Im[1, :]

i1, i2 = i1 - i1.min(), i2 - i2.min()
i1, i2 = i1 * 255. / i1.max(), i2 * 255. / i2.max()

h1=i1
h2=i2
h1 -= float(i1.min())
h1 *= 255 / h1.max()
# recover an image and scale to maximal intensity
X1 = h1.reshape(img1.shape).clip(0, 255).astype(np.uint8)

h2 -= float(i2.min())
h2 *= 255 / h2.max()

X2 = h2.reshape(img2.shape).clip(0, 255).astype(np.uint8)

cv2.imwrite('try-A1.png', X1)
cv2.imwrite('try-B2.png', X2)



