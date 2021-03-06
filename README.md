# MTCNN-TFMerge
A Merged Multi-Task CNN model.
# Introduction
This is a MTCNN model built purely by tensorflow codes, with no numpy. I integrated the three nets together and coverted the original numpy codes to the relevant tensorflow codes to build them into a whole. I did this because the official MTCNN has been trained extremely well, and we may better use it directly to detect faces. The pure tensorflow model is built in file MyDetectFace.py, and a simple demo is in the function build(). The basic codes built with numpy and tensorflow are from davidsandberg and my Tensorflow Version is 1.14.
# Update 2020.5.9
Add more codes, and now the model is able to directly output the head-cut pictures after the warp-affine process.(no need to use opencv)
# Update 2019.12.19
Now the model is able to output the transform matrix to do some warp-affine. You can use them, and together with output points, to adjust the cut faces to a more suitable angle like the demo shows below.
# Prediction Demo
I used this model to predict and get the result below.
![predict result](https://github.com/TaoKai/MTCNN-TFMerge/blob/master/stars.jpg)  
The face cuts below using the output matrix to adjust face angle.  
![predict result](https://github.com/TaoKai/MTCNN-TFMerge/blob/master/cuts.jpg)
