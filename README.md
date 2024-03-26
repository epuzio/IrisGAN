# IrisGAN

### Description
IrisGAN is a Generative Adversarial Network (GAN) built to generate images given video input.
![Milo-Output](README_images/Milo-Photo.jpg)
![Milo-Output](README_images/Milo-GAN.jpg)

### Requirements
OpenCV, TensorFlow, TensorFlow Datasets, MatplotLib, NumPy

### Installation
1. Clone the repository
2. Run 'pip install opencv-python tensorflow tensorflow-datasets matplotlib numpy'
3. To create a TFRecord:
- Convert video to image stills: `python3 detect.py load path/to/file`
- Zip images: `python3 detect.py zip`
- Create TFRecord from zip: 'python3 detect.py tfr'
3. To run the GAN:
- Run `python3 gan.py`

### Libraries Used
1. [OpenCV Cascade Classifier](https://github.com/opencv/opencv/tree/master/data/haarcascades)
2. [TensorFlow](https://www.tensorflow.org/tutorials/generative/dcgan)


### Acknowledgements
1. [TensorFlow DCGAN](https://www.tensorflow.org/tutorials/generative/dcgan)
2. [pretranied-gan-minecraft by Jeff Heaton](https://github.com/jeffheaton/pretrained-gan-minecraft)
3. [DCGAN256 by dkk](https://github.com/dkk/DCGAN256/tree/master)
4. [RectGAN by xjdeng](https://github.com/xjdeng/RectGAN/tree/master)
5. [ArtGANAE by Chee Seng Chan](https://github.com/cs-chan/ArtGAN/blob/master/ArtGAN/Genre128GANAE.py)
6. [Anime-Face-GAN by yashy3nugu](https://github.com/yashy3nugu/Anime-Face-GAN)
