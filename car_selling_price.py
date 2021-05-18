# importing all important modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# creating dataframe and fetching basic information
df = pd.read_csv('car data.csv')
# print(df.head())
# print(df.dtypes)
print(df.shape)




categorical_features = ['Seller_Type', 'Transmission', 'Owner']
for category in categorical_features:
    print(df[category].unique())

# print(df.isnull().sum())
# print(df.describe())
# print(df.columns)
final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]       # removed Car name cos it doesnt affect price

final_dataset['Current_Year'] = 2021  # to get how old the car is
print(final_dataset)

final_dataset['Car_age'] = final_dataset['Current_Year'] - final_dataset['Year']
print(final_dataset)

final_dataset.drop(['Year', 'Current_Year'], axis=1, inplace=True)
print(final_dataset)

final_dataset = pd.get_dummies(final_dataset, drop_first=True)  # first columns (cng) to prevent from dummy variable trap
# if 1 is drop then other represent it  oot tw0 are 0 , 0 then it be default the third one
print(final_dataset.head(10))

corrm = final_dataset.corr()
top_corr_features = corrm.index
plt.figure(figsize=(20, 20))


# plot heat map
g = sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap='RdYlGn')
plt.show()

# Dependent and independent features
X = final_dataset.iloc[:,1:]  # 1: means from the first columns consider all as independent festures
y = final_dataset.iloc[:,0]   # 0th is the dependent features
# print(y.head())

# ordering of the important features
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)
# Drawing this for better visualisation
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# print(X_train.shape)

from sklearn.ensemble import RandomForestRegressor
# Randomized Search CV
# Number of Trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of level in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
# MaX_depth.append(None)
#Minimum number of samples required to split a node
min_samples_split = [2, 5, 15, 100]
# Minimum number of samples required at each leaf level
min_samples_leaf = [1, 2, 5, 10]

from sklearn.model_selection import RandomizedSearchCV  # this is faster than grid cv
# create random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}
print(random_grid)

# use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()  #n_estimator = default = 100

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv= 5, verbose=2, random_state=42, n_jobs=1)
# no of iterations,   cv=cross validation, verbose=if u don't write, it won't display the training data as it shows
# n_jobs = how many Cores if 1 all cores of machine will be used
rf_random.fit(X_train, y_train)

#Predictions
predictions = rf_random.predict(X_test)
print(predictions)
sns.displot(y_test - predictions) # difference between real(actual) and predicted should be minimal hence the it looks
# closer to each other# should be like normal distributions
plt.show()
plt.scatter(y_test, predictions)  # ANd their values are linear
# plt.show()
import pickle
file = open('car_selling_price_predictions.pkl', 'wb')
pickle.dump(rf_random, file)











