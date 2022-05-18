# Kaggle G2Net Challenge Attempt
## Joshua O'Cain

### Summary

In this page, I will cover my approaches to tackling the Kaggle G2Net Gravatitational Wave challenge. The challenge provided a data set generated
by the G2Net Collaboration simulating LIGO data from the LIGO Haverford, LIGO Livingston, and VIRGO detectors with and without a detected gravitational
wave. To tackle this time series classification problem, I employ a few different signal analysis and machine learning techniques, including FFTs,
Poincare plots, and Convolutional Neural Networks.


## Gravitaional Waves

Gravitational Waves were first theorized in the 1850s but physics lacked the framework to describe them. In 1916, Albert Einstien, having formulated the general relativity field equations, predicted the existence of gravitational waves. It would be almost another century before they were finally discovered. On the morning of September 14, 2015, the Laser Interferometer Gravitational-Wave Observatories (LIGO) at Hanford, Washington and Livingston, Louisiana simoteneously observed a transient gravitational wave signal. The signal came from the collapse of binary black hole system, in which the two super massive bodies spiraled into each other and finally merged. 

![Detected Signal](https://github.com/jocain/G2Net/blob/40b4fa22e2599f24b3283af95a9f1fab6309c78f/detected%20wave.png)

The binary black hole collapse event is extremely rare, and there is a significant effort to be able to detect as many such events as possible when they occur. In October 2018, the European Cooperation in Science and Techonology initiated the G2Net collaboration with the goal of inviting collaboration among a broad range of fields in the gravitation wave pursuit, but with intentional focus on machine learning experts. In October 2021, G2Net posted a challege on Kaggle, inviting maching learning enthusiasts to join in on the search for gravitational waves. The challege asked contestants to create a machine learning model that could learn on a massive simulated data set of detected gravitational wave and noise signals from the Haverford, Livingston, and Virgo LIGO stations. In this project, I make an attempt at this challenge using a veriety of machine learning models. 

## Fourier Transforms and FFT

Fourier Transforms are based on the idea that a collection of functions can form as an orthogonal basis set, and that you
can build other functions as a linear combination of this set. In Linear Algebra, performing a dot product with of some vector with an orthonormal basis set will result in a series of weights. By using the weights in a linear combination of the basis set, the original vector can be recreated. In the same way, Fourier Transforms deconstruct functions into a power series called a frequency domain. Instead of performing dot products, the Fourier Transform utlilizes integrals over the whole range of the function to produce weights:

![Fourier](https://github.com/jocain/G2Net/blob/a6992644036e99e1b823c9fe34d6cc688140de14/ft.png)

Here, <img src="https://render.githubusercontent.com/render/math?\xi"> represents some frequency-like thing that is present in the orignial function <img src="https://render.githubusercontent.com/render/math?\f(x)"> with some weighting <img src="https://render.githubusercontent.com/render/math?\hat{f}(x)">. Note that <img src="https://render.githubusercontent.com/render/math?\xi"> and <img src="https://render.githubusercontent.com/render/math?\hat{f}(x)"> are in general complex. To reverse this process and reproduce the original function, there is also an Inverse Transform:

![Inverse Fourier](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/ift.png)

This process can also be performed with sines and cosines. This process is a generalization of the Fourier Series, which used sine and cosine waves to decompose periodic functions. Sine and Cosine Transforms also exist:

![SC Transform](https://github.com/jocain/G2Net/blob/86f377840648a0c4e06d3b5b84dffc1a8f070110/sct.png)

Here, <img src="https://render.githubusercontent.com/render/math?\lambda"> much more closely translates to frequency, and A and B become the weights, which are found by  

![Sine Transform](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/st.png)

and

![Cosine Transform](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/ct.png)

To perform this on a real dataset, we can use Fast Fourier Transforms (FFT). Because the points are descretized, there are a now limited number of avalible frequencies to investigate. The integral becomes a sum, with weights y for frequencies k, given a dataset of size N:

![FFT](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/fft.png)

And the inverse transform becomes 

![IFFT](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/ifft.png)

Similar sums exist for Discrete Sine and Cosine Transforms (DST/DCT). 

Ideally, underneath all the noise in the LIGO interferometry is some signal that consistently occurs on a frequency or a set of frequencies. If this is the case, then it's possible that we can use FFTs, DSTs, or DCTs as a method of identifying gravitational wave signals. A sample of a frequency power distribution taken for a time series of LIGO data given in the competition is shown below:

![FFT example](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/Screenshot%202022-05-06%20123809.png)

The peaks and troughs represent weights given to specific frequencies, and the magnetude of the power represents signal prevalence. This used the real part of an FFT calculation. 

## Poincare Plots

Poincare plots were a method of examining and quantifying features of signals and time series data long before computers and modern machine learning techniques could. Interestingly, these devices are named after Henri Poincare, who was also interested in gravitation waves. Poincare plots were a way of investigating self similarity process, and were usually associated with periodic functions or signals. Also known as a return map, Poincare plots are useful for identifying signals over noise. To create a Poincare plot, a time series of form <img src="https://render.githubusercontent.com/render/math?X\rightarrow(x_1, x_2,...,x_n)"> is transformed into a two-dimensional series <img src="https://render.githubusercontent.com/render/math?X'\rightarrow\{(x_1, x_2),(x_2,x_3),...,(x_{n-1}, x_n)\}"> and plotted. This creates a plot that looks like the following:

![Poincare Plot Example](https://github.com/jocain/G2Net/blob/e5932dca258e4e6b3142874578e589efd92b4178/PoincarePlot.gif)

A Poincare plot will usually develop a slanted patterd with principal axes along longest and shortest axes of the distribution. A standard way to compare Poincare plots is to take the standard deviations along these axes and devide them:

![Old Poincare Formula](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/pstat1.png)


This provides some measure of , and has been applied with great success in clinical time series analysis in the realm of cardiology. Theoretically, this study might be able to use Poincare plots to identify small gravitational signals over . A novel Poincare plot statistic has also been introduce, in which the standard deviation of is calculated

![Novel Poincare Formula](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/pstat2new.png)

This method was also investigated. 


## Convolutional Neural Networks

Convolutional Neural Networks are special variation of Artificial Neural Networks that have found great success in the ImageNet Large Scale Visual Recognition Challenge, and have since become far and away the most used tool for image classification. While images are not quite the subject of analysis here, we treat the time-dependence and the different signal sources as spatial coordinates. We already implicitly assume that there is some temporal pattern in each individual time series that can be extracted to help classification; this method transforms temporal patters into a spatial pattern to analyze the data more wholistically. CNNs have four main types of layers: Fully Connected Layers, Convolutional Layers, Pooling Layers and Flattening Layers. Fully Connected Layers are present in many types of Neural Networks. An initial set of targets <img src="https://render.githubusercontent.com/render/math?X"> undergoes a linear weighting and shifting using a weighting matrix <img src="https://render.githubusercontent.com/render/math?w"> and a bias matrix <img src="https://render.githubusercontent.com/render/math?b">. The new data will then undergo some non-linear weighting function, <img src="https://render.githubusercontent.com/render/math?a(z)">. The entire process is shown below:

![full layer process](https://github.com/jocain/G2Net/blob/1c766d723d2e262e1907dc2ecb9691c986c49711/Screenshot%202022-05-05%20121508.png)

By repeating this over mutiple layers, a neural network can theoretically produce some nonlinear relationship. Convolutional Layers work similarly, except the weighting and biasing is not performed on the whole dataset at once. Instead, a kernal of some predetermines size will slide across the data set, performing the weighting and biasing and creating something called a convoluted feature. The process with just weighting is shown below:

![Convolution Gif](https://github.com/jocain/G2Net/blob/85928c3bfd4791b1ebb5bfa79ad630df1a63c2c6/convolve.gif)

Creating mutliple convolutional layers at once can be extremely computationally expensive. To rectify this, there is usually a pooling layer after a Convolutional Layer that aggregates covoluted elements using a kerneling process, just like the convolutional layer. Aggregation process options include Max pooling or Average pooling, which takes the maximum value or average value in the kernel, respectively. After a series of convolutions and poolings, the model usually undergo a Flattening layer, in which convolved features are recombined through tiling. The result is once again two dimensional, and readable by Fully Connected Layers. The output layer is usually a Fully Connected Layer, with the output size being the number of classes in the model. The weights and biases used in the different layers are leared via a gradient decent method, minimizing a objective function called the loss. By achieving a low loss, the model learns how to produce good accuracy. 

For this project, I used a version of DenseNet optimized for time series analysis. DenseNet was a recent winner of the ImageNet Challenge, and has shown in this research to be useful for time series analysis as well. DenseNet is unique in that it employs so-called Dense blocks of convolutional, batch normalization, and pooling layers that connect to previous blocks. The structure is shown below:

![Densenet Structure](https://github.com/jocain/G2Net/blob/a7fc6d0d983ec570208f6d58a4f06e7ea35b5d76/densenet.png)

DenseNet does not employ fully connected layers beyond the output layer, so all of the image or signal processing is handled by the dense blocks. 

## Methods and Results

The data for this experiment used observations of LIGO data, each of which contained signals from three different stations. Each time series contained interferometry signals sampled for two seconds at 2048Hz, resulting in series that were 4096 points long. Training was performed using Google Colab, which allowed GPU access during the deep learning protion of this experiment. An example observation is shown below:

![Example Signal](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/Screenshot%202022-05-05%20172905.png)

The DenseNet model used 18,000 observations, were as non-deep learning models used 20,0000. This limit was a function of the space limits allowed on Colab, as well as Colab runtime limits. 

### FFT and Poincare Plots

For the heuristics-based models, the dataset was first transformed usinig the respective process before being used for training by three different types of classifiers - Logistic Regression, Support Vector Classification, and Random Forrest Classification. Though the Poincare statistics only resulted in a single heuristic per time series and were therefore computationally cheap to train classifiers with, this was not true for the Discrete Transform statistics. I kept three types of heuristics - power, frequency, and Poincare stats. For power, I kept a range of powers (40-60, depending on FFT, DCT, or SCT) around the dominant peak. For frequency, I kept only the frequency with the largest amplitude (magnitude, not highest value). For Poincare, the statistics naturally allowed only on heuristic per time series. 

Models were trained using K-Fold validation using 30 Folds, keeping the training and test accuracy and mean squared error for each fold. None of the models performed particularly well. A possible explanation is shown below:

![FFT Oveerlay](https://github.com/jocain/G2Net/blob/80b1ff8d42c64b4a6593d7f8e4f761699ec292bb/Screenshot%202022-05-06%20125222.png)

Shown in this series of graphs are the real parts of a FFT on 100 observations. From left to right are the different LIGO locations (Hanford, Livingston, Virgo). From top to bottom are the data with and with out a gravitational wave signal. In the optimal case, we would be able to clearly see a difference between corresponding top and bottom graphs, but no differences are clearly visible. The same is true when the features are scaled,

![Scaled Overlay](https://github.com/jocain/G2Net/blob/2756a35c179d943d5b9221cd4e0a4f248474d70e/scaled%20overlay.png)

and for other Discrete Transformations. Using PCA to reduce the features in the Poincare statistics to two, we can see the same problem:

![Pca overlay](https://github.com/jocain/G2Net/blob/de23cca6c8480ebfc7bb37a2e1cd6fc39b53ef92/pca%20overlay.png)

Here, detected waves are on the blue, and noise in on the orange, for a train set (left) and test set (right). Interestingly enough, many of the models overtrained, achieving as much as 0.65 accuracy on the train set, but getting less than 0.3 on the test set. Because there is no significant difference between these plots, it makes sense that it is difficult for classification algorithms to spot the differences. 

### DenseNet 

Compared to the non-deep learning models, the training of DenseNet was very non-linear. I started with a base DenseNet structure provided by that had been used in mutlivariate time series analysis of . Looking at just the loss, training of the model initally seemed fruitful. However, looking at the accuracy, the model simply refused to train.  An example of model training over 500 epochs is shown below:

![Example Learning](https://github.com/jocain/G2Net/blob/0f6598967f7578dcc262f4fe4e362a486b495d11/example%20learning.png)

As shown, the model began to stick at about a loss of 0.69315. Unfortunately, in the training possible in the time spent on this project, this seems to be the apex performance of the model. Once this performace was initially achieved, I attempted changing the learning rate. Under the assummption that the loss surface was not smooth, a learning rate kick might get the model to jump out of a local minima and get it to move towards a more global minima. Training included rate lowering if the loss plateaus, so ideally it would kick the model, and the model would return to normal training at a new, better minima. Of course, looking at a completelyh different loss might help as well, which I tried. After both were unsuccessful, I retried training from using a completely new model, freezing everything but the last convolutional and output layer, before freezing those layers and unfreezing the rest of the model. This resulted in a plateau at the same spot, as did performing the same action in reverse. I attempted adding a new fully connected layer after the Dense blocks, retrying the freezing method, but to no avail. The best model test accuracy achieved was 0.5005. This was calculated using one random 50/50 split of the data into train and test sets. 


## Conclusions

At this point, none of my models performed particularly well. While I would like to continue on this project, the most important limiting factor in my models might very well be data - not enough data to differentiate signal from noise. At this point, I am able to use about 18,000 events in the optimal case, compared to the 786,000 events provided in the challenge. The highest scoring models achieved about 80% accuracy, which means I am well below even the error that they are getting. That being said, even with the small data set I am able to use, I should still be able to improve the models that I have, and there are definitely some things I still would like to try. First, it is clear from the failure of the Discrete Transform that whatever gravitaional signal exists is small - extremely small. I want to investigate how different scalars affect the power-frequency plots, especially where the powers are very weak, as the frequencies I am looking for might still be hidden. Additonaly, I want to try and play around with the standard deviations in the Poincare plots. As a starting point, I want to try using only the standard deviation in the "skinny" direction - if the signal is indeed as small as I belive it to be, it might appear as just a small amount of noise on top of the greater Ligo Signal, which would be more visible in the SD1 statisti than the SD1 Statistic. Finally, I would like to retry building a Convolutionnal Neural Network, but begin from scratch. ThI would like to optimize my search for a good struchtre using a process like Particle Swarm or Grid searching, but I this is the most tried an true method used in this problem. I believe it can work, but it requires something that DenseNet, or possibly this form of denseNet, does not have. 

## References and Resources

[G2Net Main Page](https://www.g2net.eu/)

[G2Net Kaggle Challenge]( https://www.kaggle.com/competitions/g2net-gravitational-wave-detection)

[Gravitational Wave Detection Paper]( https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.061102)

[Gravitational Wave Article]( https://www.ligo.caltech.edu/page/gravitational-waves#:~:text=Gravitational%20waves%20are%20ripples%20in,stars%20that%20blow%20themselves%20up.)

[Fourier Transform Article]( https://en.wikipedia.org/wiki/Fourier_transform)

[Scipy Fast Fourier Transform]( https://docs.scipy.org/doc/scipy/tutorial/fft.html)

[Novel Poincare Plot Statistic Article]( https://www.researchgate.net/publication/224130464_Novel_feature_for_quantifying_temporal_variability_of_Poincare_plot_A_case_study)

[Convolutional Neural Networks Article]( https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

[DenseNet Article]( https://arxiv.org/abs/1608.06993)

DenseNet for Time Series Article
- [Paper](https://ieeexplore.ieee.org/abstract/document/9219631)
- [Github Repo]( https://github.com/josephazar/MLSTM-DenseNet)
