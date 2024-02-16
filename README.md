
## Objectives
The objective of this project is to design a minimal viable product (MVP) of a trained Convolutional neural network (CNN) model with a Graphical User Interface (GUI). The CNN-based solution aims to predict whether an individual has heart disease via analysing Electrocardiograph (ECG) images. Additionally, this project also measures the performance of the model and displays the model prediction results in the GUI.

## Running the GUI application
* Go to the GUI link and start using the GUI app. 
> https://aai-ecg-classification.streamlit.app/
* The default is the homepage.
* Press 📥Images tab on the navigation bar. Upload any ECG images taken from the ZIP file. You can upload multiple files and click on the Evaluate All button.
* After the Evaluation Completed message is shown, navigate to 📋Model Evaluation to view the prediction and evaluation results.
* Go back to the homepage and press the For Devs button located on the sidebar in the GUI to view the normal evaluation (without image augmentation).
* Reboot the GUI and refresh if there is a prompt regarding taking up too much memory
![image](https://github.com/Valdarie/AAI1001/assets/31137223/2eed49f3-4551-436a-b34e-ac6dcfcd1e99)

## Disease Classifications Categories
| Disease Classification|
| ------------- | 
|  N: Normal / Healthy Individual heart beat | 
|  S: Supraventricular premature beat | 
| V: Premature ventricular contraction |
| F: Fusion of ventricular and normal beat |
| Q: Unclassifiable beat |
| M: Myocardial infarction |

## Dataset used
The dataset used is sourced from Kaggle and comprises two collections of heartbeat signals derived from the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database.

## Project Path Directory

## Contributors
| **ROLES** | **NAME** |
| ----------| ---------|
| Codes | @Valdarie @Ashlinder |
| GUI | @Valdarie |
| Codes | @Xuanting85 | 
