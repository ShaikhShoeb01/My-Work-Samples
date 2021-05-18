import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_excel('Data_Train.xlsx')# if there are multiple sheete them provide sheets name in sheets argument
print(train_data.head())

# pd.set_option('display.max_columns', None)  # to display all columns
# print(train_data.head())

# print(train_data.info)
# print(train_data.dtypes)

train_data.dropna(inplace=True)
# print(train_data.isnull().sum())


train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
train_data["Journey_month"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.month
print(train_data.head())

# data is of 1 year only so no need to get it
# now drop date of journey
# print(train_data["Journey_day"], train_data["Journey_month"])
train_data.drop(["Date_of_Journey"], axis=1, inplace=True)
print(train_data.head())


# Similary doing for time
train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour
train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute
# print(train_data["Dep_hour"], train_data["Dep_min"])
train_data.drop(["Dep_Time"], axis=1, inplace=True)
print(train_data.head(20))

train_data["Arrival_hour"] = pd.to_datetime(train_data.Arrival_Time).dt.hour
train_data["Arrival_min"] = pd.to_datetime(train_data.Arrival_Time).dt.minute
train_data.drop(["Arrival_Time"], axis=1, inplace=True)
print(train_data.head())

# Duration e.g. 2 hr 50min not understand to model
# print(train_data['Duration'].value_counts())

duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i].strip()

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins
train_data.drop(["Duration"], axis=1, inplace=True)
print(train_data.head())



import seaborn as sns
# sns.catplot(y = "Price", x = "Airline", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
# plt.show()
    #Handling Categorical Data
# we will use oneHot coding
# but if foreign price of airline is very high then we will use LabelEncode
# print(train_data['Airline'].value_counts()))

# for airline
Airline = train_data[["Airline"]]
Airline = pd.get_dummies(train_data[["Airline"]], drop_first= True)
Airline.head()

# For Source
# sns.catplot(y="Price", x="Source", data=train_data.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3)
# plt.show()

train_data["Source"].value_counts()
Source = train_data[["Source"]]
Source = pd.get_dummies(Source, drop_first=True)

# For Destination
train_data["Destination"].value_counts()
# sns.catplot(y="Price", x="Source", data=train_data.sort_values("Price", ascending=False), kind="boxen", height=6, aspect=3)
# plt.show()

Destination = train_data[["Destination"]]
Destination = pd.get_dummies(Destination, drop_first = True)
Destination.head()


# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
train_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)
print(train_data.head())

# Now stops
print(train_data['Total_Stops'].value_counts())
# more the stops more the price, see the data...,ordinal categories so use label encoding
train_data.replace({"non-stop":0, "1 stop":1, "2 stops":2, "3 stops":3, "4 stops":4}, inplace=True)
print(train_data.head())

#concatenate df==> train_data + Airline + Source + Destination
data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)

data_train.head()

data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
data_train.head()
# print(data_train.shape)


# Why shoulhd all these are steps need to be done separately forr test data
#Because of Data leakage ...model will know the some information from the train data that leads to the overfitting

test_data = pd.read_excel("Test_set.xlsx")
print(test_data.head())

# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i].strip()

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis=1, inplace=True)

# Categorical data

print("Airline")
print("-" * 75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first=True)

print()

print("Source")
print("-" * 75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first=True)

print()

print("Destination")
print("-" * 75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first=True)

# Additional_Info contains allmost 80 % no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis=1, inplace=True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis=1)

data_test.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

print()
print()

print("Shape of test data : ", data_test.shape)

#_________________________________________Feature Selection___________________________________________________
#Some of methods are:
#heat map
#2 feature_importance_
#3 SelectKBest
print(data_train.shape)
print(data_train.columns)


X = data_train.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']] # Remove dependent featrues i.e Price   put in y dependent feature
print(X.head())

y = data_train.iloc[:, 1]
print(y.head())


# FIndinf the correletion between dependent and independent features
# For that use ExtraTreeRegressor which is used to get relation i.e which feature affection dependent i.e price
# which features are important for our output variable
# If two feature are almost related same then remove one of them...if u don't remove...u r using duplicate feauted

plt.figure(figsize= (18, 18))
sns.heatmap(train_data.corr(), annot=True, cmap="RdYlGn")
# plt.show()


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)

# Plotting important features affecting Price
plt.figure(figsize= (12, 8))
feat_importance =pd.Series(selection.feature_importances_, index=X.columns)
feat_importance.nlargest(20).plot(kind= 'barh')
plt.show()