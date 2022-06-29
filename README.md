[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7955149&assignment_repo_type=AssignmentRepo)
# Assignment-1
This is __Assignment 1__ of Seasons Of Code 2022 - Video Super Resolution.

## Problem Statement
Consider a single low-resolution image, we first upscale it to the desired size using bicubic interpolation to obtain Y. Our goal is to recover from Y, an image F(Y) that is as similar as possible to the actual high resolution image X.

-----
Helper Code has been provided. You job is to complete the code in all the files and to generate correct output.

## Step 1 - Preparing Data
There are multiple datasets which can be used. In addition to using any *appropriate* dataset on the internet, feel free to use [these](https://drive.google.com/drive/folders/1AyNme7TG-3eNuyaR9iPCqhP3-geeVY0p?usp=sharing)

There are both images for those who wish to preprocess the data and a read-made HDF5 binary file consisting of the T91 data. The goal of this step is to prepare a pytorch dataset using this (or any other) data.

## Step 2 - Bicubic Interpolation
Perform bicubic interpolation on the images to obtain poor-quality upscaling. This will act as a basis of input for the neural network.

## Step 3 - Training the Model
Create a convolutional neural network model with appropriate parameters and hyperparameters
The goal of this step is to train the neural network on the above data. Be sure to save the trained network.
Do manual tuning for the hyperparameters. This step heavily depends on how you pass data so be very careful

Include the command to run your model in run.txt

## Step 4 - Testing the Model
The goal of this step is to test the model on sample data. Use both natural images for manual comparison and automated metrics on test data for statistical comparison
Save some of the results side-by-side by optionally modifying the viewer file to visually showcase your results.

## Conclusion
Finish up the project by updating the requirements.txt and making your own observations and inferences in a README file.
Document the results and approaches used properly

Deadline:- 10th June 2022.
