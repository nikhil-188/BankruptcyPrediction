# BankruptcyPrediction
Our project aims to utilize machine learning techniques for bankruptcy prediction, leveraging data analysis and predictive models to forecast the likelihood of a company experiencing financial insolvency.

The dataset used is  Taiwan Company Bankruptcy Dataset (1999-2009)

https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction

The dataset includes financial ratios and other attributes of companies, including liquidity ratios, solvency ratios, and profitability ratios.

**Machine Learning Algorithms:**
In this research, we comparatively studied five machine learning algorithms. A brief explanation of each of them is given below

    Logistic Regression with PCA
    SVM
    Naïve Bayes
    Decision Tree
    Random Forest

**Methodology:**

First we imported the dataset Taiwan Company Bankruptcy Dataset (1999-2009), which has 6819 rows and 96 columns. Then we preprocessed the imported data by removing null values. And then we split the dataset into 70% training and 30% testing. We then applied several machine learning algorithms such as Logistic Regression with PCA, SVM, Naive Bayes, Decision Tree, and Random Forest to predict bankruptcy. The methodology begins with applying Principal Component Analysis (PCA) on the training and testing datasets to reduce the dimensionality of the data. The Logistic Regression algorithm is then used for binary classification. Regularization rate is set and Logistic Regression is fit to the training set. Predictions are made on the validation and training sets and accuracy scores are calculated. Precision and recall scores are also calculated for both the validation and training sets. Finally, an ROC curve is plotted to visualize the performance of the model. In the same way these are calculated for all the models.

For Logistic Regression with PCA, we obtained a training accuracy of 86% and validation accuracy of 87%. SVM had an accuracy of 87.6% for the validation data and 87.9% for the training data. Naive Bayes had a validation accuracy of 78.3% and a training accuracy of 76.6%.
To assess the performance of the models, we also plotted the ROC curve and calculated the AUC score. We obtained an AUC score of 0.92 for Logistic Regression with PCA, 0.928 for SVM, and 0.929 for Naive Bayes. Additionally, we plotted the partial dependency display for each algorithm to visualize the relationship between each feature and the predicted probability of bankruptcy.
Finally, we used the Decision Tree and Random Forest algorithms to perform cross-validation and obtain accuracy scores. The Decision Tree algorithm had an average cross-validation score of 0.877 and the Random Forest algorithm had an average cross-validation score of 0.879. The results of the project demonstrate the importance of using ROC curves and partial dependency curves to evaluate the performance of different machine learning models.

We also plotted the ROC curves for each algorithm and obtained an AUC score for each. Logistic Regression with PCA, SVM, and Naive Bayes had AUC scores of 0.92, 0.928, and 0.929, respectively. We also plotted the partial dependency curves for each algorithm to understand the impact of individual features on the prediction outcome.
The ROC curve is an important evaluation metric in classification problems because it measures the trade-off between the true positive rate and the false positive rate. It allows us to choose a threshold that balances the two rates, and also provides a visual representation of the model's performance. The partial dependency curves are important because they help us understand how individual features impact the model's predictions. This information can be used to identify important features, remove irrelevant ones, and improve the model's performance.

![image](https://github.com/nikhil-188/BankruptcyPrediction/assets/84719583/50457e39-621b-4a32-a947-79ef941d5d7e)

As we can see the best model among the models used is Support Vector Machine.

Partial dependency curve of SVM classifier

![image](https://github.com/nikhil-188/BankruptcyPrediction/assets/84719583/1c0e09d9-9695-4053-88a4-ee99c1d148c3)

ROC curve of SVM classifier

![image](https://github.com/nikhil-188/BankruptcyPrediction/assets/84719583/271faad9-b46f-4058-83b3-528985bde2f3)
