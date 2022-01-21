### Multi-class CNN Segmentation of 3D HAADF STEM Tomography Reconstructions of a &gamma;-Alumina/Pt catalytic material
***
Repository for the code used in " A Deep Learning Approach for Semantic Segmentation of Unbalanced Data in Electron Microscopy of Catalytic Materials".

Python scripts for :
* U-Net model
* Train model
* Evaluate model
  
  DSC, precision, recall

Training and validation data sets include ground-truth images and annotations. 
* number of classes : 3
  
  Class labels : 
  
  Background/Pores : 0,  &gamma;-Alumina : 1, Pt nanoparticles : 2
* Each images and masks are 512x512 pixels patches.

  45 for training and 15 for validation 


### Paper Link
***
Published paper can be found at:
(https://arxiv.org/abs/2201.07342)
