# A novel Fourier neural operator framework for classification of multi-sized images

**Author:** Ali Kashefi (kashefi@stanford.edu)<br>
      
**Citations** <br>
If you use the code, please cite the following journal paper: <br>

<b>#1</b> **[A novel Fourier neural operator framework for classification of multi-sized images: Application to 3D digital porous media](https://arxiv.org/abs/2402.11568)**

**Abstract** <be>

Fourier neural operators (FNOs) are invariant with respect to the size of input images, and thus images with any size can be fed into FNO-based frameworks without any modification of network architectures, in contrast to traditional convolutional neural networks (CNNs). Leveraging the advantage of FNOs, we propose a novel deep-learning framework for classifying images with varying sizes. Particularly, we simultaneously train the proposed network on multi-sized images. As a practical application, we consider the problem of predicting the label (e.g., permeability) of three-dimensional digital porous media. To construct the framework, an intuitive approach is to connect FNO layers to a classifier using adaptive max pooling. First, we show that this approach is only effective for porous media with fixed sizes, whereas it fails for porous media of varying sizes. To overcome this limitation, we introduce our approach: instead of using adaptive max pooling, we use static max pooling with the size of channel width of FNO layers. Since the channel width of the FNO layers is independent of input image size, the introduced framework can handle multi-sized images during training. We show the effectiveness of the introduced framework and compare its performance with the intuitive approach through the example of the classification of three-dimensional digital porous media of varying sizes.
