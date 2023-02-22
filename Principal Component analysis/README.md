## Principal Component Regression

Regression is performed with the data for Parkinson's disease patients for features like Unified Parkinson's Disease Rating Scale, total UPDRS value (F0 = 7), and JITTER value (F0 = 5), but with only L uncorrelated features instead of all measured features. These uncorrelated features are extracted using Principle component analysis (PCA).

Comparison is made between the results of regression with principal component analysis (PCR) and MSE regression performed in [Linear Regression](/Regression%20model%20(Parkinson's%20patients))

### Comments

This deviation of best fit line and larger error in case of PCR is due to the fact that in PCR, model is trained using vector y_hat (supervised training) which is orthonormal to feature set, unlike in MSE regression. Also, the number of features are reduced. As a result, PCR model is less overfitted to data than MSE thus having larger estimation error.

### Plots

![image](https://user-images.githubusercontent.com/25234772/220716001-d842266a-35b6-475f-b058-c9094e31e352.png)

![image](https://user-images.githubusercontent.com/25234772/220716076-a25165c8-8a3e-4834-b832-ada1f8e8d16b.png)

![image](https://user-images.githubusercontent.com/25234772/220716563-518eee71-25f8-4913-bb8c-6379ecda369e.png)

![image](https://user-images.githubusercontent.com/25234772/220716609-eee8fee1-2ffe-40af-86b7-01617ee675eb.png)
