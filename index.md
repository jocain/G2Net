# Kaggle G2Net Challenge Attempt
## Joshua O'Cain

### Summary

In this page, I will cover my approaches to tackling the Kaggle G2Net Gravatitational Wave challenge. The challenge provided a data set generated
by the G2Net Collaboration simulating LIGO data from the LIGO Haverford, LIGO Livingston, and VIRGO detectors with and without a detected gravitational
wave. To tackle this time series classification problem, I employ a few different signal analysis and machine learning techniques, including FFTs,
Poincare plots, Wavelet Transforms, and Convolutional Neural Networks.


## Gravitaional Waves

Gravitational Waves were first theorized in the 1850s but physics lacked the framework to describe them until General Relativity. Gravitational Waves were first discovered at the Laser Interferometer Gravitational-Wave Observatories (LIGO) at Hanford, Washington and Livingston, Louisiana, simoteneously. The signal came from the collapse of . 

In , the G2Net collaboration was formed with the goal of . In October 2021, the G2Net

## Fourier Transforms and FFT

Fourier Transforms are based on the idea that a collection of functions can form as an orthogonal basis set, and that you
can build other functions as a linear combination of this set. In Linear Algebra, a 

## Poincare Plots

Poincare plots were a method of examining and quantifying features of signals and time series data long before computers and modern machine learning techniques could. 

## Convolutional Neural Networks

Convolutional Neural Networks are special variation of Artificial Neural Networks commonly applied to image classification problems. While images are not quite the subject of analysis here, we treat the time-dependence and the different signal sources as spatial coordinates. We already implicitly assume that there is some temporal pattern in each individual time series that can be extracted to help classification; this method transforms temporal patters into a spatial pattern to analyze the data more wholistically. 

Convolutional Neural Networks are traditionaly 

For this project, I intend to use a version of DenseNet with the addition of Attention LSTM layers. DenseNet has already shown success in the ImageNet Large Scale Image Recognition Challenge

## Model Training

## Results

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
