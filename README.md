# funds4friends
Pipeline to estimate long-term customer engagement from short-term activivity using data from a 
platform that allows groups of friends to pool money for common purposes. These include purchasing 
gifts for common friends, paying rent, going together to a concert or a sport event, etc…
Customers can create a pool either on the company’s website or through an app. 

The datasets include user IDs, and their activity as a function of time, such as their invitations 
(sent and received), their payments, and the pools used. This dataset is used to label users as engaged 
vs non-engaged customers and to perform feature engineering. 

The main features considered are four. They account both for the number of people connected to 
the first two pools used by each user and for the time elapsed between the first two payments and the first 
two invitations (sent or received) by each user. The idea behind the computation of these features is that:
1) Engaged customers contribute to pools that are linked to more people respect to unengaged users (network effect)
2) Engaged customers use the product more frequently than unengaged users. This different behavior 
is already present from their first uses of the product.

Main files:

1) main.py: Main, interactive file to run the pipeline in python3. It calls subroutines in two files 
(all_data.py and random_forest.py). These files are used to load, merge and analyze four datasets, 
and to compute predictions using a Random Forest Classifier (using scikit-learn). 

2) all_data.py: This file contains three subroutines (merge_data, add_features, make_labels).
- merge_data: subroutine to merge data for user IDs, invitations and payments
- add_features: subroutine to compute new features from the early activity of each user
- make_labels: subroutine to label all data and output final dataset to be used for Machine Learning predictions 

3) ml_algo.py: This file computes predictions using a Random Forest Classifier, together with multiple summary 
statisitics such as accuracy, precision, recall, F1 score, etc...
