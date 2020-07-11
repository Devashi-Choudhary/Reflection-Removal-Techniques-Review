# Reflection Removal

The image of an object can vary dramatically depending on lighting, specularities/reflections, and shadows. Thus, removing undesired reflections from images is of great importance in computer vision and image processing. It means to enhance the image quality for aesthetic purposes as well as to preprocess images in machine learning and pattern recognition applications. Thus, the goal of the project is to review various techniques that are used for removing reflection from the image.

# Dependencies

opencv==4.2.0

keras==2.3.1

tensorflow>=1.15.2

imutils==0.5.3

numpy==1.18.2

matplotlib==3.2.1

argparse==1.1 

pandas==0.23.4

scipy==1.1.0

MATLAB

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

5. **Reflection Removal using Deep Learning :** 

# Results 
