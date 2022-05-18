# Kaggle G2Net Challenge Attempt
## Joshua O'Cain

### Summary

In this page, I will cover my approaches to tackling the Kaggle G2Net Gravatitational Wave challenge. The challenge provided a data set generated
by the G2Net Collaboration simulating LIGO data from the LIGO Haverford, LIGO Livingston, and VIRGO detectors with and without a detected gravitational
wave. To tackle this time series classification problem, I employ a few different signal analysis and machine learning techniques, including FFTs,
Poincare plots, Wavelet Transforms, and Convolutional Neural Networks.


## Gravitaional Waves

Gravitational Waves were first theorized in the 1850s but physics lacked the framework to describe them. In 1916, Albert Einstien, having formulated the general relativity field equations, predicted the existence of gravitational waves. It would be almost another century before they were finally discovered. On the morning of September 14, 2015, the Laser Interferometer Gravitational-Wave Observatories (LIGO) at Hanford, Washington and Livingston, Louisiana simoteneously observed a transient gravitational wave signal. The signal came from the collapse of binary black hole system, in which the two super massive bodies spiraled into each other and finally merged. 

Insert image of detected signal

In October 2018, the European Cooperation in Science and Techonology G2Net collaboration was formed with the goal of . In October 2021, the G2Net

## Fourier Transforms and FFT

Fourier Transforms are based on the idea that a collection of functions can form as an orthogonal basis set, and that you
can build other functions as a linear combination of this set. In Linear Algebra, performing a dot product with of some vector with an orthonormal basis set will result in a series of weights. By using the weights in a linear combination of the basis set, the original vector can be recreated. In the same way, Fourier Transforms deconstruct functions into a power series called a frequency domain. Instead of performing dot products, the Fourier Transform utlilizes integrals over the whole range of the function to produce weights:

$$ \hat{f}(\xi) = \int^{infty}_{-\infty}{f(x)e^{-2i\pi x \xi}dx $$

Here, <img src="https://render.githubusercontent.com/render/math?\xi"> represents some frequency-like thing that is present in the orignial function <img src="https://render.githubusercontent.com/render/math?\f(x)"> with some weighting <img src="https://render.githubusercontent.com/render/math?\hat{f}(x)">. Note that <img src="https://render.githubusercontent.com/render/math?\xi"> and <img src="https://render.githubusercontent.com/render/math?\hat{f}(x)"> are in general complex To reverse this process and reproduce the original function, there is also an Inverse Transform:

$$ f(x) = \int^{infty}_{-\infty}{\hat{f}(\xi)e^{2i\pi x \xi}dx $$

This process can also be performed with sines and cosines. This process is a generalization of the Fourier Series, which used sine and cosine waves to decompose periodic functions. Sine and Cosine Transforms also exist:

$$ f(x) = \int^{infty}_{-\infty}{f(x)(A(\lambda)\cos(2\pi\lambda x) + B(\lambda)\sin(2\pi\lambda x))dx $$

Here, <img src="https://render.githubusercontent.com/render/math?\lambda"> much more closely translates to frequency, and A and B become the weights, which are found by  

$$ f(x) = \int^{infty}_{-\infty}{f(x)(A(\lambda)\cos(2\pi\lambda x) + B(\lambda)\sin(2\pi\lambda x))dx $$

and


$$ f(x) = \int^{infty}_{-\infty}{f(x)(A(\lambda)\cos(2\pi\lambda x) + B(\lambda)\sin(2\pi\lambda x))dx $$.

To perform this on a real dataset, we can use Fast Fourier Transforms (FFT). Because the points are descretized, there are a now limited number of avalible frequencies to investigate. The integral becomes a sum, with weights y for frequencies k, given a dataset of size N:

$$ y(k) = \sum^{N-1}_{n = 0} e^{-2i\pi kn/N}x_n $$

And the inverse transform becomes 

$$ x(n) = \frac{1}{N}\sum^{N-1}_{k = 0} e^{2i\pi kn/N}y(k) $$

Similar sums exist for Discrete Sine and Cosine Transforms (DST/DCT). 

Ideally, underneath all the noise in the LIGO interferometry is some signal that consistently occurs on a frequency or a set of frequencies. If this is the case, then it's possible that we can use FFTs, DSTs, or DCTs as a method of identifying gravitational wave signals. 

## Poincare Plots

Poincare plots were a method of examining and quantifying features of signals and time series data long before computers and modern machine learning techniques could. Interestingly, these devices are named after Henri Poincare, who was also interested in gravitation waves. Poincare plots were a way of investigating self similarity process, and were usually associated with periodic functions or signals. Also known as a return map, Poincare plots are useful for identifying signals over noise. To create a Poincare plot, a time series of form <img src="https://render.githubusercontent.com/render/math?X\rightarrow(x_1, x_2,...,x_n)"> is transformed into a two-dimensional series <img src="https://render.githubusercontent.com/render/math?X'\rightarrow\{(x_1, x_2),(x_2,x_3),...,(x_{n-1}, x_n)\}"> and plotted. This creates a plot that looks like the following:

Insert example Poincare Plot

A Poincare plot will usually develop a slanted patterd with principal axes along longest and shortest axes of the distribution. A standard way to compare Poincare plots is to take the standard deviations along these axes and devide them:

old poincare formula

This provides some measure of , and has been applied with great success in clinical time series analysis in the realm of cardiology. Theoretically, this study might be able to use Poincare plots to identify small gravitational signals over . A novel Poincare plot statistic has also been introduce, in which the standard deviation of is calculated

novel poincare formula

This method was also investigated. 


## Convolutional Neural Networks

Convolutional Neural Networks are special variation of Artificial Neural Networks that have found great success in the ImageNet Large Scale Visual Recognition Challenge, and have since become far and away the most used tool for image classification. While images are not quite the subject of analysis here, we treat the time-dependence and the different signal sources as spatial coordinates. We already implicitly assume that there is some temporal pattern in each individual time series that can be extracted to help classification; this method transforms temporal patters into a spatial pattern to analyze the data more wholistically. CNNs have four main types of layers: Fully Connected Layers, Convolutional Layers, Pooling Layers and Flattening Layers. Fully Connected Layers are present in many types of Neural Networks. An initial set of targets <img src="https://render.githubusercontent.com/render/math?X"> undergoes a linear weighting and shifting using a weighting matrix <img src="https://render.githubusercontent.com/render/math?w"> and a bias matrix <img src="https://render.githubusercontent.com/render/math?b">. The new data will then undergo some non-linear weighting function, <img src="https://render.githubusercontent.com/render/math?a(z)">. The entire process is shown below:

full layer process

By repeating this over mutiple layers, a neural network can theoretically produce some nonlinear relationship. Convolutional Layers work similarly, except the weighting and biasing is not performed on the whole dataset at once. Instead, a kernal of some predetermines size will slide across the data set, performing the weighting and biasing and creating something called a convoluted feature. The process with just weighting is shown below:

Like the fully connected layer, the entire set of convoluted features undergoes some non-linear activation function. A couple choices of activation functions are shown below:

activation functions

Creating mutliple convolutional layers at once can be extremely computationally expensive. To rectify this, there is usually a pooling layer after a Convolutional Layer that aggregates covoluted elements using a kerneling process, just like the convolutional layer. Aggregation process options include Max pooling or Average pooling, which takes the maximum value or average value in the kernel, respectively. After a series of convolutions and poolings, the 

Convolutional Neural Networks are traditionaly 

For this project, I used a version of DenseNet optimized for time series analysis. DenseNet was a recent winner of the ImageNet Challenge, and has shown in this research to be useful for time series analysis as well. DenseNet is unique in that it employs so-called Dense blocks of convolutional, batch normalization, and pooling layers that connect to previous blocks. The structure is shown below:

Densenet picture

DenseNet does not employ fully connected layers beyond the output layer, so all of the image or signal processing is handled by the dense blocks. 

## Methods and Results

This data used 18,000 observations of LIGO data, each of which contained signals from three different stations. 

### FFT and Poincare Plots

For the non-deep learning stage, FFT, FST, FCT, and Poincare Plot statist, the dataset was first transformed

### DenseNet 

## Conclusions

## References and Resources

G2Net Main Page

G2Net Kaggle Challenge

Gravitational Wave Detection Paper

Gravitational Wave Article

Fourier Transform Article

Scipy Fast Fourier Transform

Poincare Plot Article

Novel Poincare Plot Statistic Article

Convolutional Neural Networks Article

DenseNet Article

DenseNet for Time Series Article
