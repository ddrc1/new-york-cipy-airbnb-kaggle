import sys
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == '__main__':
    args = sys.argv[1:]
    path = args[0]
    min_samples_split = int(args[1])
    min_samples_leaf = int(args[2])
    bootstrap = bool(args[3])
    max_features = args[4]
    try:
        max_features = float(max_features)
    except:
        pass

    n_estimators = int(args[5])
    n_jobs = int(args[6])

    # Read data
    df = pd.read_csv(path)

    # Pre-processing
    df.drop(columns=['name', 'host_name', 'id', 'host_id'], inplace=True)
    df['last_review'] = pd.to_datetime(df['last_review'])
    last_day = df['last_review'].max()
    df['days_since_last_review'] = df['last_review'].apply(lambda x: (last_day - x).days)
    df.dropna(inplace=True)

    df = pd.concat([df, pd.get_dummies(df['neighbourhood_group']), pd.get_dummies(df['neighbourhood']), pd.get_dummies(df['room_type'])], axis=1)
    df.drop(columns=['neighbourhood_group', 'neighbourhood', 'room_type', 'last_review'], inplace=True)
    
    # Train
    x = df.drop(columns='price')
    y = df['price'].ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=10)

    model = RandomForestRegressor(criterion='absolute_error', min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                  max_features=max_features, n_estimators=n_estimators, n_jobs=n_jobs)
    model.fit(x_train, y_train.ravel())

    # Validation
    y_pred = model.predict(x_test)

    metrics = {'rmse': mean_squared_error(y_test, y_pred) ** (1/2), 'mae': mean_absolute_error(y_test, y_pred), 'r2': r2_score(y_test, y_pred)}

    mlflow.log_metrics(metrics)
    print("metrics:", metrics)

    mlflow.sklearn.log_model(model, "rf")