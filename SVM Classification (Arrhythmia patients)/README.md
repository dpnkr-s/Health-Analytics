## Support Vector Machine

Same data for arrhythmia patients in [classification](/Minimum%20Distance%20and%20Bayesian%20classification%20(Arrythmia%20patients)) is used to implement Support Vector Machine (SVM) classifier. SVM classification is performed using two types of kernels, ‘Linear’ and ‘Gaussian’ kernels and then comparison is made between the performances of both kernels.

![image](https://user-images.githubusercontent.com/25234772/220736994-2a990328-bc05-4823-8585-e1e9276247f3.png)

In the boxplot of classloss values for different kernels, it can be safely concluded that Gaussian kernel outperformed Linear kernel having the lowest classloss value, also having consistently lower classloss values over a range of box constraint values.

Also, looking at elapsed time to perform classification using each kernel below, gaussian kernel performed better than linear kernel again with less time rewuired for computation.
- Mean elapsed time for Linear kernel: **5.42s**
- Mean elapsed time for Gaussian kernel: **2.17s**
