This project is a web app that accepts a image from user and identify the lung area if the user input is a chest xray. If the user input is not a chest xray then the same input will be printed as output.

To acheive this two models have been trained
1. Binary classification model to identify if a given image is chest xray or not (chest_xray_binary.ipynb)
2. Model to identify lung area from a chest xray (lung_segmentation.ipynb)

To train the binary classification model a random dataset with a folder of chest xrays and another folder with non chest xrays is been taken and trained using MobileNetV2.
To train lung segmentation model Unet architecture is used. The dataset considered have 800 images of chest xrays and 704 out of them have mask images. Jaccard and Dice coefficients are used to calculate accuracy for the model.

Lung segmentation model is build with reference to the below github repository. The link of the dataset is available in readme of the below mentioned repository.
https://github.com/IlliaOvcharenko/lung-segmentation

Web app is build with reference to https://github.com/am1tyadav/Image-Classifier-Web-App

