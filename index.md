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

The data for this experiment used observations of LIGO data, each of which contained signals from three different stations. Each time series contained interferometry signals sampled for two seconds at 2048Hz, resulting in series that were 4096 points long. Training was performed using Google Colab, which allowed GPU access during the deep learning protion of this experiment. 

### FFT and Poincare Plots

For the heuristics-based models, the dataset was first transformed usinig the respective process before being used for training by three different types of classifiers - Logistic Regression, Support Vector Classification, and Random Forrest Classification. This process was repeated with feature standardization using standard scaling between the heurisitc calculation and model training. Though the Poincare statistics only resulted in a single heuristic per time series and were therefore computationally cheap to train classifiers with, this was not true for the Discrete Transform statistics. A table containing the number of kept features per statistic is included below. 

kept stats

Models were trained using K-Fold validation using 30 Folds, keeping the training and test accuracy for each fold. The results are shown tabulated:

kept stats

As is clear from the above table, none of the models performed particularly well. A possible explanation is shown below:

Shown in this series of graphs are the real parts of a FFT on 6,000 observations. From left to right are the different LIGO locations (Hanford, Livingston, Virgo). From top to bottom are the data with and with out a gravitational wave signal. In the optimal case, we would be able to clearly see a difference between corresponding top and bottom graphs, but no differences are clearly visible. The same is true when the features are scaled,



and for other Discrete Transformations. Using PCA to reduce the features in the Poincare statistics to two, we can see the same problem:


Here, detected waves are on the left, and noise in on the right. Because there is no significant difference between these plots, it makes sense that it is difficult for classification algorithms to spot the differences. 

### DenseNet 

Compared to the non-deep learning models, the training of DenseNet was very non-linear. I started with a base DenseNet structure provided by that had been used in mutlivariate time series analysis of . Training of the model initally seemed fruitful. An example of model training over 500 epochs is shown below:

As shown, the model began to stick at about a loss of 0.69315. Unfortunately, in the training possible in the time spent on this project, this seems to be the apex performance of the model. Once this performace was initially achieved, I attempted changing the learning rate. Under the assummption that the loss surface was not smooth, a learning rate kick might get the model to jump out of a local minima and get it to move towards a more global minima. Training included rate lowering if the loss plateaus, so ideally it would kick the model, and the model would return to normal training at a new, better minima. After this was unsuccessful, I retried training from using a completely new model, freezing everything but the last convolutional and output layer, before freezing those layers and unfreezing the rest of the model. This resulted in a plateau at the same spot, as did performing the same action in reverse. I attempted adding a new fully connected layer after the Dense blocks, retrying the freezing method, but to no avail. The best model statistics achieved are shown below:



This was calculated using one random 50/50 split of the data into train and test sets. 


## Conclusions

At this point, none of my models performed particularly well. While I would like to continue on this project, the most important limiting factor in my models might very well be data - not enough data to differentiate signal from noise. At this point, I am able to use about 18,000 events in the optimal case, compared to the 786,000 events provided in the challenge. The highest scoring models achieved about 80% accuracy, which means I am well below even the error that they are getting. That being said, even with the small data set I am able to use, I should still be able to improve the models that I have, and there are definitely some things I still would like to try. First, it is clear from the failure of the Discrete Transform that whatever gravitaional signal exists is small - extremely small. I want to investigate how different scalars affect the power-frequency plots, especially where the powers are very weak, as the frequencies I am looking for might still be hidden. Additonaly, I want to try and play around with the standard deviations in the Poincare plots. As a starting point, I want to try using only the standard deviation in the "skinny" direction - if the signal is indeed as small as I belive it to be, it might appear as just a small amount of noise on top of the greater Ligo Signal, which would be more visible in the SD1 statisti than the SD1 Statistic. Finally, I would like to retry building a Convolutionnal Neural Network, but begin from scratch. ThI would like to optimize my search for a good struchtre using a process like Particle Swarm or Grid searching, but I this is the most tried an true method used in this problem. I believe it can work, but it requires something that DenseNet, or possibly this form of denseNet, does not have. 

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
