# Supervised Learning — Algorithm Comparison

> Portfolio analysis notes converted from prior report drafts. Author bylines and course identifiers removed.

Supervised Learning
 
Introduction
Supervised learning (SL) is a subcomponent of machine learning where a program learns a function that maps an input to an output based on defined input-output pairs. In other words, when training supervised learning models, the model is told that the answer to X1 is Y1, X2 is Y2,… This then allows the model to predict, for example, that the answer to X203 is Y203. 
SL can be used for both classification and regression problems. In this project, SL algorithms support vector machines, neural networks, decision tree, boosted decision tree, and k-nearest neighbor will be used to analyze and predict outcomes from breast cancer and Titanic data. These two classification problems are exceptionally interesting both together and apart, are non-trivial, and allow for comparisons and analysis of the various algorithms. 
Breast cancer is the one of the most common types of cancers in females and more than 200,000 cases occur per a year in the US. This dataset will include thirty different body measurements (X variables) from a digitized image of a fine needle aspirate of a breast mass tied to if the resultant tissue diagnosis (Y) was malignant or benign. The dataset contains of 569 datapoints.
The Titanic dataset includes gender, age, family size, and characteristics about where they likely were on the ship tied to if they survived the ships sinking. In total, the Titanic data has 12 input (X) variables, and 891 datapoints (rows). The breast cancer and Titanic binary classification datasets will allow us to compare how the same algorithms can produce very different results depending on the datasets. Additionally, if the measurements prove to be effective at predicting breast cancer or cruise ship survival, it can form the start of a larger project that applies machine learning in the important pursuit of saving lives.
General Methodology
To conduct these analyses, the datasets were preprocessed - meaning string variables were modified into numerical categorical variables and input values were scaled. This was done to improve results by minimizing impacts of skewness and to convert the data into numerical values which the algorithms understand. 
Then, the data was split into a training and test set were 20% was dedicated to the test set. Next, Python scikit-learn packages were used with parameters tweaked to find the optimal performance. Hyperparameter tuning was carried out using grid search and 10-fold cross-validation.  Grid search is an exhaustive search tuning technique that tries to find the optimum values of hyperparameters which are parameters that are not trained by the training set and that impact the accuracy of the model. In the following sections, the performance of each model, the hyperparameter values, and key learnings will be discussed. The models are first analyzed individually and then compared at the end.
Support Vector Machines (SVM)
SVM is a frequently preferred algorithm as it typically produces high accuracy with a low required computational power. It works by trying to find a hyperplane that distinctly classifies the data points in an N-dimensional space, where N is the number of features. It looks for the plane that has the maximum distance between both classes which thereby provides some reinforcement that future data points can be classified with more confidence.
The kernels in SVM are functions used to transform data so that more problems can be solved using SVM, such as those with a large number of dimensions. A kernel helps form the hyperplane in the higher dimensions without raising the complexity and they help solve non-linear problems with the help of linear classifiers. Kernels can also help solve overfitting. For this project, linear, radial basis function (rbf), and sigmoid kernels were tested. Additionally, polynomial kernels were tested but did not complete within 15 minutes, so was removed from the code for time. C, a regularization parameter, was tested for 0.1, 1, 10 and gamma, a kernel parameter, was tested for 0.1, 1, 10, 100 in the grid search. Gamma adjusts how similar two points must be to be considered “similar”.
For both datasets, the training data performed better than the test data except on sigmoid. Sigmoid is a common kernel used in neural nets and is represented as . For Titanic, rbf had the best performance as determined by the F1 score for both test and training sets and thus was used for the subsequent metrics and graphs. For breast cancer, linear was used because the test data got worse while the training data got better on rbf, which shows overfitting. The RBF family of kernels follow the form:  and linear is .

For breast cancer and Titanic, the F1 training was higher than the cross-validation but there was overlap when considering the standard deviations. Possible values for F1 score range from 0 to 1. This test also showed the standard deviation decreased greatly until the number of training examples reached a larger number.

As expected, training time increased as the number of training examples increased. However, it is beneficial that prediction time stayed relatively constant and was a magnitude lower than the training time.

Since Titanic had more datapoints overall, it was expected to have a larger training time. However, despite having more data, these initial models have the breast cancer SVM model performing better on all indicators. The experiment also showed that gamma can have a major impact on the test set results. Additionally, both training and cross-validation scores increase as C, which controls regularization, increases. This occurs up to a point until cross-validation starts to decrease as training increases thereby indicating the start of overfitting.

Breast Cancer
Titanic
Best Parameters
C: 10, Gamma: 1 (Linear)
C: 1, Gamma: 0.1 (RBF)
Recall
0.85
0.61
Precision
1
0.71
Accuracy
0.95
0.75
AUC
0.93
0.73
F1 Score
0.92
0.66
Training Time [s]
0.00399
0.01169
Prediction Time [s]
0.00039
0.00136
Neural Networks
Neural networks are inspired by the brain and can handle both supervised and unsupervised learning. For this algorithm, learning rates of 0.01, 0.05, and 0.1 were tested along with hidden layer sizes of 5, 25, 50, and 100. The scikit-learn multi-level perceptron classifier with an Adam solver and logistic activation parameter were used.
The graphical results showed that increasing the number of hidden layers helps initially but then has negligible impact. This also matches the results shown in the table were the grid search said that the best number of hidden layers for the breast cancer was 5 and for Titanic was 50.

As with SVM and as expected, training F1 score is above the cross-validation and the standard deviation decreases with more training examples. Additionally, as before the Titanic F1 score is lower than the breast cancer score and there is a larger difference between training and cross-validation in the Titanic dataset than there is in the breast cancer dataset.

The training time for both datasets is much higher than it was for SVM. However, similar to SVM, prediction time is considerably lower than training time and prediction time is relatively constant.

Both datasets had an optimal learning rate of 0.1. The lower the learning rate the slower the model travels along the slope to find a minima, but this can also help by increasing the chance that the model doesn’t overshoot the minimum and fail to converge. Perfect learning rates are a balance between accuracy, convergence, and time. These datasets were relatively small, but if the dataset was millions of datapoints from advertising, for example, it could take hours or days depending on computational power.
The size of the datasets can also be a determinate of the learning rate as the smaller breast cancer dataset can allow for a smaller learning rate while still having a reasonable runtime when processing all the data. 

Breast Cancer
Titanic
Best Parameters
Hidden Layer Sizes: 5 Learning Rate: 0.1
Hidden Layer Sizes: 50 Learning Rate: 0.1
Recall
0.93
0.70
Precision
1.00
0.77
Accuracy
0.97
0.80
AUC
0.96
0.78
F1 Score
0.96
0.73
Training Time [s]
0.16315
0.55065
Prediction Time [s]
0.00015
0.0046

Decision Tree
Decision trees continuously split data until an outcome can be predicted by using nodes, edges/branches, and leaf nodes. This segment tested various min_sample_leaf and max_depths to achieve pruning-like results. Pruning is a data compression technique that reduces the size of decision trees by removing sections of the tree that are non-critical and redundant when classifying instances and is done to reduce overfitting. Decision trees also have a “criterion” which is a function to measure the quality of a split. Two types are Gini and entropy. For this project, entropy was used as the quality function/criterion.
The max tree depths impact on F1 score diminished in both the test and training sets.

The decision tree learning curve graphs of F1 score by number of training examples is similar in style to the other algorithms’ graphs.

The decision trees had the most interesting performance thus far regarding the time. For both datasets, up-to-a-point, training time was below prediction time, but as the number of training examples increased training time increased above prediction time. This occurred around 20-30% of the total datasets size for breast cancer and Titanic.

In the models, pruning decreased the test error rate by preventing overfitting.

Breast Cancer
Titanic
Best Parameters
Max Depth: 3 Min Samples Leaf: 2
Max Depth: 6 Min Samples Leaf: 6
Recall
0.85
0.64
Precision
0.97
0.79
Accuracy
0.94
0.79
AUC
0.92
0.76
F1 Score
0.91
0.70
Training Time [s]
0.00167
0.00129
Prediction Time [s]
0.00012
0.00012

Boosted Decision Tree
Two common types of boosting are AdaBoost and gradient boosting. This section focuses on gradient boosting. The boosted decision tree follows similar logic to the non-boosted decision tree, but cutoff thresholds can be more aggressive/lower since boosting combines multiple learners. The number of estimators and learning rate are also introduced as hyperparameters which will determine the contribution of each tree classifier.

The breast cancer graph had horrible performance at a low number of training examples which is different from previous algorithms or the boosted Titanic algorithm.

Like the decision tree, the Boosted’s time gradually increased with more training examples.

Interestingly, the boosted decision tree performed similarly to the non-boosted. (This is further discussed in the conclusion.) Some metrics were slightly better and some were slightly worse. The high value for n_estimators means there is high number of weak learners.

Breast Cancer
Titanic
Best Parameters
Learning Rate: 0.1 Max Depth: 2 Min Samples Leaf: 23 N_Estimators: 102
Learning Rate: 0.0505 Max Depth: 3 Min Samples Leaf: 4 N_Estimators: 102
Recall
0.82
0.65
Precision
1
0.78
Accuracy
0.94
0.79
AUC
0.91
0.77
F1 Score
0.9
0.71
Training Time [s]
0.08080
0.07974
Prediction Time [s]
0.00041
0.00053

K-Nearest Neighbors (KNN)
KNN is a non-parametric method that groups similar points based on distance and the number of allowed neighbors. This experiment used Euclidean distance and varied the hyperparameter n_neighbors from 1 to 200. The result was 18 neighbors for the breast cancer data and 10 for Titanic.
As the number of neighbors increased, the F1 score for both the test and train sets and for both datasets decreased. This is expected because increasing the number of neighbors requires the model to generalize more and form “arbitrary matches” on non-relevant datapoints.

Next, the F1 score by number of training examples increased and then plateaued which is not too different from any of the other algorithms.

However, unlike all the other algorithms, kNN always had a prediction time greater than a training time. This tells that the model can quickly assign existing data into groups but struggles to assign new data into existing groups.

Breast Cancer
Titanic
Recall
0.70
0.51
Precision
0.97
0.74
Accuracy
0.89
0.74
AUC
0.84
0.70
F1 Score
0.81
0.60
Training Time [s]
0.00146
0.00083
Prediction Time [s]
0.10966
0.11023

Comparison
The neural net training time was considerably longer than the other models. Second was boosted tree. However, the neural network training also showed a lot more variability in time as the number of training examples increased.

KNN had significantly higher prediction time – between 0.10 s and 0.12 s – and thus was removed from the graph to better show the comparisons of the other models. The graphs show that decision tree is fairly consistent on timing regardless of the training example count. This is because the trees do not change during prediction and no more additional calculations or branch formations occur; thus, allowing for quick assigning of new datapoints.

These final graphs show that F1 score can vary greatly between models and datasets and that while a sufficient number of training examples is important, many other variables go into the model that will impact its F1 score.

Conclusion
The project showed applying supervised machine learning models to real-life examples. However, reaching a level where a business would implement a model would likely take more data, testing, and optimization. For instance, the project gives a stronger intuition on the complexities that impact a model’s performance, such as hyperparameters, and what makes it far more difficult to achieve stellar results, such as data quality.
For example, large number of classes and the highly clustered data that does not split easily into a particular class makes it difficult for models to generalize. Additionally, there must be sufficient amount of data. A general rule of thumb is a magnitude more than the number of features to prevent curses of dimensionality. These two datasets had a good number of datapoints but the breast cancer could benefit from more. Next, data cleaning and scaling impacts the results. The breast cancer x-values were scaled before being inputted into the models. As a result, it achieved better results on every model than it initially would have. Lastly given a data set with a feasible size, cross-validation will certainly help the learning.
For the experiments, on F1 Score, the neural network performed the best and KNN performed worst. However, KNN had the best training time and the worst prediction time. Neural nets had the worst training time and decision trees had the best prediction time. Querying a decision tree involves tree traversal and with tree depths low, querying can be fast. KNN inference is the slowest because it involves distance computation with respect to each point, but training is faster since it simply involves data storing. Neural net training is slow due to gradient computations in backpropagation. 
Neural networks can also be very sensitive to learning rates and their regularization parameter. Neural nets and decision trees are eager learners. KNN is a lazy learner.
Boosting fared poorly in these comparisons likely because a decision tree is not a weaker learner. A better result could have occurred with a decision stump. Also, boosting takes on characteristics of the algorithm being boosted so metrics and time can vary greatly depending on what algorithm is used.
SVM also performed well. Adjusting the SVM kernel makes a significant difference in performance especially with a dataset containing a large number of attributes. When the data is not linearly separable, experimenting with different kernel settings can help identify which kernel works best. Also, while SVM didn’t have the worst time, SVM can take a long time with larger datasets. Initially, a 40,000+ datapoint set was used but replaced with a smaller dataset because models were taking too long to run.
Regarding how much performance was due to the problems chosen, the datasets do have an important impact but preprocessing and algorithms have a huge impact as the performance between the two datasets and between each model varied greatly on F1 score. These steps and decisions are what cause the results to be what was achieved.
References
Titanic - machine learning from disaster. (n.d.). Retrieved February 19, 2021, from https://www.kaggle.com/c/titanic
Lbronchal. (2017, November 21). Breast cancer dataset analysis. Retrieved February 19, 2021, from https://www.kaggle.com/lbronchal/breast-cancer-dataset-analysis
