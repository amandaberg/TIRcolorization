# Generating Visible Spectrum Images from Thermal Infrared

This repo contains the TIR2Lab model and weights presented in the PBVS2018 paper with the same name:

A. Berg, J. Ahlberg, and M. Felsberg, Generating Visible Spectrum Images from Thermal Infrared, 2018 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), (2018)

OBS! This is the same repo that previously resided here: https://gitlab.ida.liu.se/amabe60/PBVS2018 

## Load the model
Example of how to load the model and pre-trained weights in Keras 2.0:

```
with open('TIR2Lab_model.json') as f:
        json_string = f.readline()
transformer = model_from_json(json_string)
transformer.load_weights('TIR2Lab_weights.h5')
```

Please note that input images should be in range [0,1] and that the output images will be in a normalized Lab color space. Use the following function (postprocess_tir2lab_results) to convert an output image from the nomalized Lab color space to RGB:

```
import numpy as np
from PIL import Image
from skimage import color

def postprocess_tir2lab_results(batch, validationPath):		
	rescaled_batch = lab_rescale(batch)
	for idx, image in enumerate(rescaled_batch):
		image = color.lab2rgb(image.astype('float64'))
		image = image*255
		im = Image.fromarray(np.squeeze(image).astype('uint8'))
		im.save(validationPath + '/I000' + str(idx) + '.png')
		
def lab_rescale(im):
    im[:,:,:,0] = im[:,:,:,0]*100.0
    im[:,:,:,1] = im[:,:,:,1]*185.0 - 87.0
    im[:,:,:,2] = im[:,:,:,2]*203.0 - 108.0
    return im
```

## Source code
Since many have asked, I have now uploaded the source code as well. Be aware that even though it has been organized and cleaned, it is still in a kind of "development" stage. Use at your own risk!
