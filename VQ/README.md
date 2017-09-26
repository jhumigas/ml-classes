# Vector Quantization

This small lab is about vector quantization techniques, a class of unsupervised learning algorithms commonly used for clustering.
Among their advantages, one can notice is that they keep the topology of the initial dataset. 
In Vector Quantization, we consider a set of prototypes, each one averaging in some sort of way 
information of a certain set of samples. To compare a given prototype to the sample it is representing, we define a loss function, that introduces on a global scale a distorsion i.e the expectation, when a sample x are taken according to X, of the error made by assimilating x to its closest prototype in Ω [1]. The core idea of the vector quantization is to minimize that distorsion which involves chosing the right number of prototypes and placing them right.

The original lab subject can be found [here](http://www.metz.supelec.fr//metz/personnel/frezza/ApprentissageNumerique/TP-MachineLearning/NonSupervise.html)
In this lab, we consider images of handwritten digits and the main objective is to build a self organizing map of prototypes.
Base.h is the source code to use the database of handwritten digits. The Handwritten digits come from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/)

One first thing to note is the choice of the distance. The euclidian distance is not the most appropriate distance to use here. 
In fact, starting from an example of simple black and white digits, the euclidian distance will be the same for the following samples:
*TODO : Add example*

The inconvenience of using only the euclidian distance, is that it does not really keep the ressemblance information between two images.
A better measure should, say, implement the correlation between two digits.


## References

Jeremy Fix, H.F Buet, M. Geist and F. Pennerath, [Machine Learning](http://sirien.metz.supelec.fr/spip.php?article91)