# new-york-city-airbnb-kaggle
Kaggle project

Objectives:
- What can we learn about different hosts and areas?
- What can we learn from predictions? (ex: locations, prices, reviews, etc)
- Which hosts are the busiest and why?

### How to reproduce using MLflow:
mlflow run . -e main -P min_samples_split=4 min_samples_leaf=2 bootstrap=False max_features='sqrt' n_estimators=300 n_jobs=-1 </br>
It will run a Random Forest algoritm</br>
If you desire to use different parameters, feel free to change the values

link: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
