## Classification

Data for arrhythmia patients is used to perform classification in MATLAB, initially into two categories ('healthy' and 'arrhythmic') and then into all 16 levels of arrhythmia as classified by doctors. Classification is first performed using *Minimum distance criterion* and then using *Bayesian criterion*.

Results of classification are presented as probabilities of true/false positives and probabilities of true/false negative to further analyze the quality of classification in the case of medical data. For example, probability for true positives is calculated as,

<img width="555" alt="Screenshot 2023-02-22 at 11 42 30 PM" src="https://user-images.githubusercontent.com/25234772/220718726-56a42c34-bcb7-46f9-8430-1555880af19c.png">

Results are thus displayed in the following format,

<img width="535" alt="Screenshot 2023-02-22 at 11 44 51 PM" src="https://user-images.githubusercontent.com/25234772/220719641-d9d56fb1-1758-4139-aeb3-44970f5e617b.png">

### Results of classification

#### Minimum Distance criterion

<img width="535" alt="Screenshot 2023-02-22 at 11 45 10 PM" src="https://user-images.githubusercontent.com/25234772/220720002-2fb717af-77ed-47a2-b54f-7b3df76cafb3.png">

#### Bayesian criterion

<img width="535" alt="Screenshot 2023-02-22 at 11 45 30 PM" src="https://user-images.githubusercontent.com/25234772/220720126-68526cdb-c3ea-4c03-bd6a-efb7ee43e66b.png">

### Comments

Probabilities obtained from Bayesian Criterion - solution I (considering that probabilities of occurrence of each class is different) are quite similar to probabilities obtained from Minimum Distance Criterion (with PCA). Bayesian criterion has slightly improved test specificity, this could be attributed to inclusion of a-priori probabilities of occurrence of each class which is given in dataset.

A significant improvement in classification quality is noticed after using Bayesian criteria – solution II (considering that variance of hypothesis for each class is also different). This results in very high sensitivity and specificity while using this classifier as a test to detect arrhythmia.

It is observed that while classifying data into 16 classes, Minimum Distance Criterion, with PCA, outperformed every other solution and has highest overall probabilities for True Positives and True Negatives for all classes. This could be attributed to complex decision regions in this method coupled with lack of large enough dataset, which is not available from source repository.
