# NegBinFit
Negative binomial fit with Frank Copula

It uses two negative binomial PMFs in the form:

![equation](https://latex.codecogs.com/gif.latex?{\displaystyle&space;\Pr(X=k)={\frac&space;{\Gamma&space;(r&plus;k)}{k!\,\Gamma&space;(r)}}\left({\frac&space;{r}{r&plus;m}}\right)^{r}\left({\frac&space;{m}{r&plus;m}}\right)^{k}\quad&space;{\text{for&space;}}k=0,1,2,\dotsc&space;})

where the mean is m and the variance ![equation](https://latex.codecogs.com/gif.latex?m&space;&plus;&space;\frac{m^2}{r}) coupled by the Frank copula in the form:

![equation](https://latex.codecogs.com/gif.latex?-{\frac&space;{1}{\theta&space;}}\log&space;\left[1&plus;{\frac&space;{(\exp(-\theta&space;u)-1)(\exp(-\theta&space;v)-1)}{\exp(-\theta&space;)-1}}\right])

where u and v are two negative binomial CDFs, to fit a provided numerical probability distribution i.e. a normalized bidimensional histogram. It gives back the two means and r parameters of the marginals, as well as the parameter ![equation](https://latex.codecogs.com/gif.latex?\theta).

In the main of the py file there is a working example (requires matplotlib).

Requirements:
- Python 3.5 or higher
- Scipy
- Numpy
- MatPlotLib (to visualize the example).
