## K-means Clustering

K-means clustering algorithm is implemented to cluster data for arrhythmia patients used in [classification](/Minimum%20Distance%20and%20Bayesian%20classification%20(Arrythmia%20patients)). Two variants of K-means algorithm, hard and soft K-means, are implemented and analyzed in MATLAB.

### Hard K-means, 2 clusters

Starting with mean vector obtained from classification in [classification](/Minimum%20Distance%20and%20Bayesian%20classification%20(Arrythmia%20patients)) as initial vectors. Following plots are obtained to make comparison between the initial and the final vectors.

Hard K-means decisions matched with doctors’ decision: **57.52**%

![image](https://user-images.githubusercontent.com/25234772/220724132-7f13b662-9dd1-4e66-ab55-4ce4970d4c5a.png)

### Soft K-means, 2 clusters

Starting with random vectors as initial vectors and equal initial probability for each cluster. Following plots are obtained to make comparison between mean vector obtained from classification and the final vectors.

Soft K-means decisions matched with doctors’ decision: **66.59**%

![image](https://user-images.githubusercontent.com/25234772/220724388-990549c1-7a44-4cbf-b8f9-c615f9327ef2.png)
