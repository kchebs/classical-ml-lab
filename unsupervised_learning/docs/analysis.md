# Unsupervised Learning & Dimensionality Reduction

> Portfolio analysis notes converted from prior report drafts. Author bylines and course identifiers removed.

Proj. 3: Unsupervised Learning & Dimensionality Reduction
				
Introduction
The purpose of this project is to use dimensionality reduction and clustering (unsupervised learning) to preprocess data which trains a supervised learning neural network. The two clustering algorithms used were k-means clustering and expectation maximization. The four dimensionality reduction algorithms explored were: principal component analysis (PCA), independent component analysis (ICA), randomized projections (RP) and, as the method of choice, random forest feature selection. All algorithms were implemented in Python using scikit-learn. The countless experiments were run on a breast cancer dataset and titanic dataset – these are the same datasets as in the supervised learning module. When appropriate, a 20% training size was used.
Breast cancer (BC) is one of the most common types of cancers in females and more than 200,000 cases occur per a year in the US. This dataset will include thirty different body measurements (X variables) from a digitized image of a fine needle aspirate of a breast mass tied to if the resultant tissue diagnosis (Y) was malignant or benign. The dataset contains of 569 datapoints.
The Titanic (TT) dataset includes gender, age, family size, and characteristics about where people were likely on the ship tied to if they survived the ships sinking. In total, the Titanic data has 12 input (X) variables, and 891 datapoints (rows). The breast cancer and Titanic binary classification datasets will allow us to compare how the same algorithms can produce very different results depending on the datasets. Additionally, if the measurements prove to be effective at predicting breast cancer or cruise ship survival, it can form the start of a larger project that applies machine learning in the important pursuit of saving lives. Both datasets do not have data evenly distributed amount features or classes – about 37% were malignant in the BC and 38% survived in TT.
Clustering: K-Means and Gaussian Mixture EM
K-Means clustering splits a dataset into k clusters. This is done through an itterative process where the centroid is moved to minimize the distance between each instance and the centroid of the assigned cluster. This is commonly called “within cluster sum of squared errors” and is apart of one of two popular methods to determine the appropriate number of clusters for a dataset. The other popular method is using the average silhouette score. In this experiment, k was varied from 2 to half the dataset size which means that the largest tested would have a minimum of 2 data points in a cluster. Reviewing both Silhoutte scores and elbow method sum of squared error graphs, the cluster counts were chosen (reported in table below).

Different distance methods and sub-algorithms can be used such as Chebyshec, Minkowski, Euclidean, and Manhattan for distance and Lloyd and Elkan for algorithm. This K-Means uses  Elkan algorithm with Euclidean distance.

The silhouette coefficient can range from -1 to 1. It gives the average value for all the samples at that cluster count and gives a perspective into the density and separation of the formed clusters. A score closest to 1 is ideal because it indicated that samples are far away from neighbor clusters.
When varying values in the experiment, increasing the number of clusters did increase the accuracy but eventually will lead to overfitting. Also, the benefit of more clusters will gradually diminish, which is the concept behind the cluster count vs sum of squared errors graph below.
 

For the breast cancer dataset, while the max silhouette score occurred at 2 clusters and there are two (malignant and benign) classes, a lower sum of squared errors and higher homogeneity score and higher F1 score was achieved at higher cluster counts (24). Titanic had higher F1 and homogeneity scores at high cluster counts too (100). Lastly, as observed in the plots, the elbow location isn’t clear for both datasets. This can sometimes indicate that K-means is not the best algorithm to find clusters.

Silhouette coefficient scores shouldn’t be used to evaluate GMM/EM performance because they work well only on spherical clusters and GMMs do not necessarily produce spherical clusters. K-Means can be seen as a special case of GMM with equal covariance per component.

The expectation maximization uses a gaussian mixture. It is an algorithm that finds k distributions of data such that the likelihood of data given distributions is maximized. It does this by alternating between estimating the log-likelihood for the current estimates for the parameters and maximizing the likelihood found on the expectation step. The likelihood is often represented as a log to handle large and small values the same way and is based on the probability density of the function.
The breast cancer EM performed worse on all metrics except recall and training time. 15 was chosen as the number of distributions even though the silhouette score was highest at 7 because 15 achieved a lower AIC and BIC and achieved a higher F1 score amount other metrics. Titanic EM performed worse than Titanic KM on all metrics. 

The model complexity curves for the breast cancer and titanic dataset are very different because the BIC curve starts to drastically increase for the breast cancer data over the tested distributions. This made it easier to find a max threshold of acceptable distribution values as you want the BIC and AIC to be as low as possible. The Titanic dataset has both BIC and AIC progressively decrease but looks like it may possibly start to increase at the end. These differences likely occurred because the Titanic dataset is a lot more complex with factors that are more independent and uncorrelated that could affect someone’s survival.

For this project, the number of clusters and random seed were varied. Both data sets’ K-Means achieved better results than EM. Instinctively, the minimum number of clusters should match the number of class outputs or features. However, neither had the overall best parameters at the number of output classes (2 – benign and malignant or survived and died) or at the features * output classes (30 * 2 for BC and 12 * 2 for TT). This means the classes and features are getting divided further for the Titanic dataset and do not need to be as divided for the BC. Cross-validation was not used in this section but could potentially further improve the results. However, preprocessing scaling was conducted on the BC dataset because K-Means is susceptible to features with larger ranges which in turn would dominate the optimization problem.

Breast Cancer - KM
Breast Cancer - EM 
Titanic - KM
Titanic - EM
Best Parameters
Clusters: 24
Distributions: 15
Clusters: 100
Distributions: 97
Recall
0.90
0.90
0.71
0.68
Precision
0.92
0.82
0.77
0.75
Accuracy
0.93
0.89
0.81
0.79
AUC
0.92
0.89
0.79
0.77
F1 Score
0.91
0.86
0.74
0.71
Training Time [s]
0.04
0.03
0.08
0.07
No. Iterations to Converge
12
25
6
14
Log-Likelihood Lower Bound
N/A
112.08
N/A
47.73

Dimensionality Reduction (DR): PCA, ICA, RP/RCA, RFC
Dimensionality reduction is the process of decreasing the number of random variables to a series of principal variables. Within the concept of DR are feature selection and extraction. Selection works by finding a subset of the original variables. Extraction transforms the data into a lower dimensional space. The three methods assigned for this project (PCA, ICA, and RP) fall into the feature extraction category, but the random forest (RF) classifier approach is feature selection.
PCA maps the data to linear planes in such a way that maximizes variance. This looks like individual variables are re-organized into linear combinations of variables and creates a matrix where the eigenvalues covariance is maximized. ICA approaches the data such that the variables are the output of many unobserved sources like multiple sources recorded from a microphone. RP is like PCA in that the data is projected into lower dimensions, however the directions of projection are independent of the data and serve as an efficient way to reduce high dimensional data while preserving distances between instances. RF calculates each features importance. If a feature falls outside the 95% cumulative sum, then it is dropped.
Both the BC and TT datasets had similar graphs. In the graphs, explained variance is the amount of variance described by each of the selected components – thus the higher the number the better. Explained variance can be calculated as a ratio of related eigenvalue and sum of eigenvalues of all eigenvectors which are found as part of the eigen decomposition of the transformation matrix, and which is a covariance matrix in the case of principal component analysis. These eigenvectors represent the principal components that contain most of the information represented using independent variables.
In sklearn, you can either set the number of components or use a cutoff ratio like 95% or ‘mle’. All methods were tested in this experiment, and in the end, 6 was chosen for Titanic PCA based on the graph below and 5 for BC as these achieve about 95% of the cumulative explained variance. For ICA, it was 7 for BC and 5 for TT – this is because there are more features for the breast cancer dataset than the titanic dataset. For RP, it was 11 for BC and 8 for TT.

The above PCA graph also shows that most of the variance is explained by the first few components.	
Clustering on dimensionality Reduction
Using the above found values for PCA, ICA, and RP; clusters were run on these transformation algorithms and with the random forest feature selection.

The above five graphs show the similar shape of EM homogeneity scores for Titanic across all dimensionality reduction algorithms and the original.

The general shapes of the elbow, silhouette, homogeneity, and F1 K-Means plots with dimensionality reduction were similar across the x-axis to the ones without dimensionality reduction. For the y-axis, some plots achieved slightly lower values in BC for PCA, but the differences aren’t obvious on the graph. The training time graphs were different.
For ICA, the elbow graph shape was similar but started at a much lower sum of squared distances value. The silhouette score graph was also different – achieving a plateau much more suddenly, but the training time graph was similar unlike the PCA graph. The RCA graphs were similar but more jagged which is likely due to the randomness of RCA. The RFC graphs were also similar. Due to the similarity of the graphs, the cluster counts were also about the same even though the clusters obtained after PCA are different. The similar cluster counts are likely due to the algorithms having to perform minimal dimensionality reduction. However, the Titanic dataset had less similar clusters pre- and post- dimensionality reduction than the breast cancer dataset had pre- and post. ICA performed worse on most metrics while RFC was best on BC and RCA was best on TT.
K-Means
Breast Cancer - PCA
Breast Cancer - ICA 
Breast Cancer - RCA
Breast Cancer - RFC
Titanic - PCA
Titanic - ICA
Titanic - RCA
Titanic - RFC
Clusters
24
24
24
24
100
100
100
100
Recall
0.80
0.70
0.79
0.81
0.60
0.56
0.74
0,69
Precision
0.94
0.84
0.93
0.93
0.73
0.75
0.78
0.76
Accuracy
0.91
0.84
0.90
0.91
0.76
0.76
0.82
0.80
AUC
0.88
0.81
0.88
0.89
0.73
0.72
0.80
0.78
F1 Score
0.86
0.76
0.85
0.87
0.66
0.64
0.76
0.72
Training Time [s]
0.04
0.03
0.02
0.02
1.22
0.34
0.34
0.31
No. Iterations to Converge
16
14
12
9
6
4
4
7
GMM may be better where k-means fails since it is a soft clustering algorithm. For EM/GMM, many of the clustering with feature extraction graphs were similar in shape to the ones without feature extraction for both datasets. Yet, the RFC feature selection had the largest differences out of all the graphs. I believe this is a combination from the selection-attribute, i.e., dropping features, and of its randomness – these two in combination changed breast cancer BIC line from its original U-shape to a downward trending curve in the BC clustering with dimensionality reduction.
When reviewing all the graphs, small changes in the distribution counts could occur but given they are small differences and looking holistically; the distribution count was left at the original number. This balances silhouette score with F1 score, AIC, and BIC. It also allows for easier comparison in the table below.
EM
Breast Cancer - PCA
Breast Cancer - ICA 
Breast Cancer - RCA
Breast Cancer - RFC
Titanic - PCA
Titanic - ICA
Titanic - RCA
Titanic - RFC
Distributions
15
15
15
15
97
97
97
97
Recall
0.68
0.83
0.82
0.81
0.58
0.55
0.65
0.73
Precision
0.97
0.68
0.95
0.95
0.73
0.77
0.77
0.74
Accuracy
0.87
0.79
0.92
0.91
0.76
0.76
0.79
0.80
AUC
0.83
0.80
0.90
0.89
0.73
0.72
0.77
0.79
F1 Score
0.80
0.75
0.88
0.87
0.65
0.64
0.71
0.74
Training Time [s]
0.03
0.03
0.02
0.01
0.07
0.08
0.05
0.07
No. Iterations to Converge
23
36
26
11
15
19
10
18
Log-likelihood Lower Bound
-4.75
23.88
7.87
-26.27
1.13
18.56
23.60
31.94
Likewise, ICA performed the worse on most metrics with RFC for TT and RCA for BC performing the best on most metrics especially F1 score. ICA tends to perform better when there is strong independence. All algorithms were similar on training speeds, but you can easily notice larger datasets had larger times.
Again, although the cluster count is the same, the clusters obtained after PCA, ICA, and RCA are different because the data is in a lower-dimensional space and GMM clustering depends on spatial distribution of data.
Neural Net with Dimensionality Reduction
The neural network was trained on the breast cancer dataset which has 569 instances and 30 features. 5 hidden layers were used, which is the same as in 
The training and prediction times for all the models were similar except RFC’s training time which was much quicker. This is likely due to RFC dropping features it deems unimportant. RCA/RP had the best recall, accuracy, F1 score, and AUC. The data in RP are projected randomly into a new space of different directions. Thus, this could help with overfitting and the different runs have generally the same trend but vary with about two percent depending on the run and seed location.
  

ICA did worse which is expected since it also performed worse above in the non-neural net section. I believe this might be due to the innate fact that independence which is key to ICA is limited in health data because, for example, you can’t have half a healthy heart and organs work dependently.
Lastly, as expected training neural net times was longer than non-neural net times.

Breast Cancer - Original
Breast Cancer - PCA 
Breast Cancer - ICA
Breast Cancer - RCA
Breast Cancer – RFC
Recall
0.88
0.79
0.74
0.94
0.88
Precision
1
0.94
0.88
0.97
0.84
Accuracy
0.96
0.91
0.88
0.97
0.89
AUC
0.94
0.88
0.84
0.97
0.89
F1 Score
0.93
0.86
0.80
0.96
0.86
Training Time [s]
0.13958
0.13339
0.14064
0.13622
0.05298
Prediction Time [s]
0.00014
0.00013
0.00013
0.00013
0.00012

Neural Net with Clustering and Dimensionality Reduction
A learning rate of 0.01, 5 hidden layers, and a test size of 0.20 was used in the neural nets construction. 5-fold cross validation was conducted.

For the neural net with clustering and dimensionality reduction, ICA did not achieve the worse and the original dataset was consistently one of the best. However, RFC did score the best in the comparison run below and RCA was the worst. The randomness of these algorithms played a major role this time as some runs fell into local optimums. Since one run of the Breast Cancer – RFC had a 0 for recall, precision, and F1 Score, I further investigated by running the Titanic dataset. The Titanic dataset achieved a 0.77 recall, 0.75 precision, and 0.76 F1 score. I also achieved different values when I ran the breast cancer again. This shows the impact randomness can have on these types of algorithms and the importance of setting a seed if one wants to achieve consistent results.

Breast Cancer - Original
Breast Cancer - PCA 
Breast Cancer - ICA
Breast Cancer - RCA
Breast Cancer - RFC
Recall
0.86
0.77
0.67
0.66
0.95
Precision
0.88
1
1
0.88
0.86
Accuracy
0.90
0.91
0.88
0.87
0.93
AUC
0.90
0.89
0.84
0.81
0.93
F1 Score
0.87
0.87
0.81
0.75
0.90
Training Time [s]
0.14161
0.08628
0.13064
0.16523
0.09873
Prediction Time [s]
0.00012
0.00011
0.00011
0.00013
0.00018

Conclusion
Clustering is a useful pre-processing technique that organizes data in an unsupervised manner. Dimensionality reduction assists in removing non-useful features to improve clustering and learning accuracy. However, different algorithms work better on different types of data. EM clustering is based on probabilities; thus, binary attributes will have high probabilities per classification as compared with numeric features. EM is also more impacted by seed location than K-Means. Additionally, PCA and ICA require numeric or binary attributes so nominal features will have to be converted through one-hot encoding. This has the potential to greatly increase the number of features and therefore computation time. Thus, some of the performance throughout the experiment was due to the problems and algorithms chosen. Also, larger datasets would have larger training times as experienced when comparing titanic to breast cancer or the larger neural net with cluster labels to the smaller neural net without cluster label feature. 
In this experiment, the clusters did not perfectly line up with the binary classifications and the projection axes for ICA did not capture meaningful information. This is because ICA decomposes a multivariate signal into independent components and requires nongaussianity. Strict independence is difficult in health with co-worker pieces and the decisions of 1912 that created the Titanic dataset do not follow clean Gaussian distributions.
Some changes to the algorithms that might be made or explored in future experiments can be a different cutoff threshold in RFC and further exploring different distance methods – Euclidean adds weight to the distance of outliers while Manhattan is more evenly weighted. Additionally, possible future work can also be on comparing the algorithms on adjusted mutual information which measures the agreement between two assignments ignoring permutations from 0 to 1.
Appendix

 
References
Titanic - machine learning from disaster. (n.d.). Retrieved February 19, 2021, from https://www.kaggle.com/c/titanic
Lbronchal. (2017, November 21). Breast cancer dataset analysis. Retrieved February 19, 2021, from https://www.kaggle.com/lbronchal/breast-cancer-dataset-analysis
