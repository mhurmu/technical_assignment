import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import itertools
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

sales_data = pd.read_csv('bakery_sales_2021-2022.csv')
coding_data = pd.read_csv('Bakery coding.csv')
weather_data = pd.read_csv('weather_2021.01.01-2022.10.31.csv')

sales_df = pd.DataFrame(sales_data)
coding_df = pd.DataFrame(coding_data)
weather_df = pd.DataFrame(weather_data)

sales_data['time'] = pd.to_datetime(sales_data['time']).dt.time
sales_data['datetime'] = pd.to_datetime(sales_data['date'].astype(str) + ' ' + sales_data['time'].astype(str))
opening_times = sales_data.groupby('date')['datetime'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)
opening_times_df = opening_times.reset_index(name='opening_time_hours')

weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df['snow'] = weather_df['snow'].fillna(0)
weather_df.drop(columns=['tsun'], inplace=True)

merged_df = pd.merge(sales_df, coding_df, on='Article', how='left')
merged_df['Adjusted_Quantity'] = merged_df['Quantity'] * merged_df['Number']
merged_df['date'] = pd.to_datetime(merged_df['date'])

aggregated_df = merged_df.groupby(['date', 'Group'])['Adjusted_Quantity'].sum().reset_index()

pivot_df = aggregated_df.pivot(index='date', columns='Group', values='Adjusted_Quantity').fillna(0)

complete_df = pd.merge(pivot_df, weather_df, on='date', how='left')
opening_times_df['date'] = pd.to_datetime(opening_times_df['date'])
complete_df['date'] = pd.to_datetime(complete_df['date'])
complete_df = pd.merge(complete_df, opening_times_df, on="date", how='left')

complete_df['day_of_week'] = complete_df['date'].dt.dayofweek
complete_df['month'] = complete_df['date'].dt.month

complete_df['closed_ahead'] = 0
complete_df['closed_behind'] = 0
date_set = set(complete_df['date'].dt.date)

for i, entry in complete_df.iterrows():
    count = 0
    if i == (len(complete_df) - 1):
        continue
    while (entry['date'] + pd.Timedelta(days=count+1)).date() not in date_set:
        count += 1
    complete_df.at[i, 'closed_ahead'] = count

for i, entry in complete_df.iterrows():
    count = 0
    if i == 0:
        continue
    while (entry['date'] - pd.Timedelta(days=count+1)).date() not in date_set:
        count += 1
    complete_df.at[i, 'closed_behind'] = count

print(complete_df.groupby("day_of_week")["closed_ahead"].mean())
print(complete_df.groupby("day_of_week")["closed_behind"].mean())
print(complete_df.groupby("day_of_week")["opening_time_hours"].mean())

if ~(pivot_df >= 0).all().all():
    raise "Not all entries are non-negative!"

groups = ['bread', 'cake', 'drink', 'packaging', 'pastry', 'prepared_plate', 'sandwich', 'sweet']
predictions = []

for group in groups:
    current_df = complete_df.copy()
    lags = [-2, -1, 1]
    prod_lags = [-2, -1]

    lagged_weather = ['tavg', 'prcp', 'snow', 'wspd', 'tmax']
    for feature in lagged_weather:
        for lag in lags:
            current_df[f'{feature}_lag{lag}'] = current_df[feature].shift(-lag)
    for lag in prod_lags:
        current_df[f'{group}_lag{lag}'] = current_df[group].shift(-lag)
    indep_vars = [*[weather+"_lag"+str(lag) for weather, lag in itertools.product(lagged_weather, lags)],
                  *[group+"_lag"+str(lag) for lag in prod_lags], *lagged_weather, 'day_of_week', 'month', 'closed_ahead', 'closed_behind']

    X = current_df[indep_vars]
    y = current_df[group]

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = RandomForestRegressor(random_state=42, **{'bootstrap': True, 'max_depth': 10, 'max_features': 1.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100})

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    predictions.append(y_pred)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Mean Absolute Percentage Error:", mape)

    importances = model.feature_importances_
    feature_names = X.columns

    if group in ["bread", "pastry"]:
        plt.figure(figsize=(10, 6))
        plt.plot(complete_df['date'], complete_df[group], label='Actual units sold')
        plt.plot(complete_df['date'][X_train.shape[0]:], y_pred, label='Predicted units', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel(f'Total {group} sold')
        plt.title(f'Graph of sales of {group}')
        plt.legend()
        plt.show()

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    top_features = importance_df['Feature'].head(6).tolist()
    print(top_features)

summed_predictions = np.sum(predictions, axis=0)

train_size = int(0.8 * len(complete_df))
plt.figure(figsize=(10, 6))
plt.plot(complete_df['date'], complete_df[groups].sum(axis=1), label='Actual items sold')
plt.plot(complete_df['date'][train_size:], summed_predictions, label='Predicted items', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Total items sold')
plt.title('Graph of total items sold')
plt.legend()
plt.show()

mse = mean_squared_error(complete_df[groups].sum(axis=1)[train_size:], summed_predictions)
mae = mean_absolute_error(complete_df[groups].sum(axis=1)[train_size:], summed_predictions)
mape = mean_absolute_percentage_error(complete_df[groups].sum(axis=1)[train_size:], summed_predictions)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Mean Absolute Percentage Error:", mape)
