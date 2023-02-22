## Hierarchical Clustering and Classification

### Hierarchical clustering

Hierarchical clustering is implemented in MATLAB on dataset of **chronic kidney disease** (CKD) patients to identify two clusters of patients who either *'have ckd'* or *'dont have ckd'*. Important features are identified from decision tree obtained after classification.

Cluster tree or dendrogram, and the cluster assignments for ‘i-th’ row as obtained after using MATLAB functions *pdist, linkage, cluster*. Plot for tree is presented below,

<img width="600" alt="image" src="https://user-images.githubusercontent.com/25234772/220725755-03c6536b-7d6a-46a8-8373-a6ad711aff96.png">

After comparing the result of clustering obtained from cluster tree with the decisions given by doctors, the following table of error probabilities is obtained,

<img width="600" alt="Screenshot 2023-02-23 at 12 12 04 AM" src="https://user-images.githubusercontent.com/25234772/220727145-d8b0fe16-0b3a-4146-a89b-5e81267a7cb5.png">

### Hierarchical classification

Classification is performed using MATLAB function *fitctree* and the following decision tree is obtained as a result,

<img width="600" alt="image" src="https://user-images.githubusercontent.com/25234772/220728298-fc2fe621-78ab-437c-89b9-6246d32f6a83.png">

After considering the most important features and their values obtained from the tree above, a new classification is made and the error probabilities obtained after comparing new classification with the decisions made by doctors. Error probabilities are presented in a table below

<img width="600" alt="Screenshot 2023-02-23 at 12 17 56 AM" src="https://user-images.githubusercontent.com/25234772/220728953-e3ba90a8-1bb7-4deb-9bf6-aa12c13f6440.png">

### Comments

From the decision tree obtained after *fitctree* classification, it is known that feature number 15, or **Blood Haemoglobin levels (in g/dL)** is the most important feature deciding whether or not patient suffers from CKD. Patients are most likely to have CKD when their Blood Haemoglobin levels are below 13.05 and **Packed cell volume** is below 44.5. Otherwise, if patients’ Blood haemoglobin level is above 13.05, then those with **Specific gravity** < 1.0175 are likely to have CKD; and if both Blood Haemoglobin and Specific gravity are above the previously mentioned values then patients with **Albumin** above 0.5 are likely to have CKD.
