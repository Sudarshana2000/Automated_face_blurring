# Automated_face_blurring

Nowadays, face recognition is of utmost importance in almost every industry. But what if the situation demands its opposite? For privacy, identity protection and security, anonymising face by blurring it helps a lot.


This requirement can be accomplished by using OpenCV and Python.

## Two ways

- Plain blurring
- Pixelized blurring

### Plain blurring

- Perform face detection
- Extract face ROI
- Apply Gaussian Blur
- Replace the blurred face in original image / video frame

### Pixelated face blurring

This is more demanding and pleasant to use.
- Detect and extract face ROI
- Divide it into MXN blocks
- Compute mean RGB pixel intensity of each block
- Annotate a rectangle on the block to create "pixelated" effect
- Apply the processed face ROI in original image

## Samples

<div style="float:left">
<div style="float:left"><img width="45%" src="https://github.com/Sudarshana2000/Automated_face_blurring/blob/master/images/input1.jpg" />
<img width="45%" src="https://github.com/Sudarshana2000/Automated_face_blurring/blob/master/images/output1.jpg" />
</div>
<br /><br />


![video](videos/output3.gif)