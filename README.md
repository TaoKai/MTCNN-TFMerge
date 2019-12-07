# MTCNN-TFMerge
A Merged Multi-Task CNN model.
# Introduction
This is a MTCNN model built purely by tensorflow codes, with no numpy. I integrated the three nets together and coverted the original numpy codes to the relevant tensorflow codes to build them into a whole. I did this because the official MTCNN has been trained extremely well, and we may better use it directly to detect faces. The pure tensorflow model is built in file MyDetectFace.py, and a simple demo is in the function build(). The basic codes built with numpy and tensorflow are from davidsandberg.
