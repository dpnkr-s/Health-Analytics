## Linear Regression

Linear regression is performed on the given data of Parkinson's disease patients in MATLAB. Features like Unified Parkinson’s Disease Rating Scale (UPDRS:  total UPDRS value and Jitter value), are expensive to measure and are estimated by regressing them over other features less expensive to measure. 

Three different algorithms to solve linear regression problem are implemented and comparison is made among the obtained results. Different methods being implemented are as follows,

- Mean Square Error (MSE)
- Gradient Descent Algorithm
- Steepest Descent Algorithm

### Comments

While working with sufficiently large databases it is expected for MSE to take longest time as it needs to compute matrix inversion, and Steepest descent to take the least amount of time between them. 

As in the case of train data, all the methods have also fitted test data very well with very small error. Overall, regression models obtained from all methods are satisfactory and give very similar results.

### Plots

![image](https://user-images.githubusercontent.com/25234772/220712789-68274337-a77c-4473-8488-04fa34d8521a.png)

![image](https://user-images.githubusercontent.com/25234772/220712836-b630ece4-9ad9-444d-96eb-b8ac55b8366a.png)

![image](https://user-images.githubusercontent.com/25234772/220712906-20f9d2f5-6eb0-4750-9e78-dae9c2e04053.png)
