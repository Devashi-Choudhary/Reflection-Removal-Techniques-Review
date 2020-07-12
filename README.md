# Reflection Removal

The image of an object can vary dramatically depending on lighting, specularities/reflections, and shadows. Thus, removing undesired reflections from images is of great importance in computer vision and image processing. It means to enhance the image quality for aesthetic purposes as well as to preprocess images in machine learning and pattern recognition applications. Thus, the goal of the project is to review various techniques that are used for removing reflection from the image.

![GIF](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review/blob/master/Readme_Images/20200712_224304.gif)


# Dependencies
python

opencv

Pytorch (torch & torchvision)

imutils

matplotlib

argparse

numpy

skimage

tqdm

pandas

scipy

MATLAB

# Dataset

The sample image used for testing is provided by [SIR2 benchmark dataset](https://sir2data.github.io/).

# How to execute code

1. **Averaging :** For given set of images, we perform averaging for reflection removal.

Open the Averaging folder and run `python Averaging.py -i 5_images_lowers` where -i is path to folder which contains set of images. 

2. **Independent Component Analysis :**   Based on concept of reflection of light and independence of underlying distribution in reflected image, two images captured at different polarising orientation are taken and solved for:   Y=MX  , where Y = [y1, y2] , two images ,  M is mixing matrix [a,b; c,d] (each denoting the amount of reflection) and  X= [x1,x2] are painting and reflecting components in the two images.

Open the ICA folder and run `python ICA.py -i1 1.png -i2 2.png` where -i1 and -i2 are path to input images.

**Note :** For next two techniques, you need MATLAB software.

3. **Relative Smoothness :**  It models the input image as a linear combination of transmission and smooth reflection layer,  I = T + R, where R is a smooth function.Based on sparsity prior of natural scenes, the objective function is to maximize the probability of joint distribution of T and R. 

You can go [here](https://github.com/yyhz76/reflectSuppress) and run `reflection_removal.m ` 

4. **Reflection Suppression via Convex Optimization :** Convex model is used to suppress the reflection from a single input image. It implies a partial differential equation with gradient thresholding, which is solved efficiently using Discrete Cosine Transform. 

You can go [here](https://github.com/alexch1/ImageProcessing) and run `reflecSuppress.m `

5. **Reflection Removal using Deep Learning :** Single-image reflection removal method based on generative adversarial networks. 

Open the Deep_Learning folder and run `python GANs.py --path /input ` where --path path to input dataset that contains reflection images.

# Results & Limitations

1. **Averaging :** We have done averaging by taking 5,10 and 20 number of input images.

![average](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review/blob/master/Readme_Images/averaging.JPG)

**Limitations :** Averaging requires set of images to remove reflection.

2. **Independent Component Analysis :** It takes two images as input. As in this method, the images is a mixture of some ratio two images and our goal to separate the two images. We have generated the images for testing.

![ICA](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review/blob/master/Readme_Images/ICA.JPG)

**Limitations :** The techniques employed here, work under their underlying assumptions, ICA requires two same layers being at different polarisation angles, which is a rare phenomenon. 

3.  **Relative Smoothness :** Images from SIR2 benchmark dataset are used to remove reflection.

![RS](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review/blob/master/Readme_Images/Relative_Smoothness.JPG)

**Limitations :** Relative smoothness removes reflections which are smooth in nature.

4. **Reflection Suppression via Convex Optimization :** 

![RS](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review/blob/master/Readme_Images/reflect_Suppress.JPG)

**Limitations :** Reflective suppress tries to suppress reflection, as parameter h increases the image tends to more blur.

5. **Reflection Removal using Deep Learning :** 

![DL](https://github.com/Devashi-Choudhary/Reflection-Removal-Techniques-Review/blob/master/Readme_Images/GANs.JPG)

# Contributors

[Arti Singh](https://github.com/Arti2512)

[Neha Goyal](https://github.com/Neha-16)

# References

1. [Separating reflections and lighting using independent components analysis.](https://dspace.mit.edu/bitstream/handle/1721.1/6675/AIM-1647.pdf?sequence...)

2. [Single image layer separation using relative smoothness.](https://openaccess.thecvf.com/content_cvpr_2014/papers/Li_Single_Image_Layer_2014_CVPR_paper.pdf)

3. [Fast Single Image Reflection Suppression via Convex Optimization.](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Fast_Single_Image_Reflection_Suppression_via_Convex_Optimization_CVPR_2019_paper.pdf)

4. [Single Image Reflection Removal Based on GAN With Gradient Constraint.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8868089) & [Implementation](https://github.com/ryo-abiko/GCNet)
