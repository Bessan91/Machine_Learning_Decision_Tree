
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# reading file
df = pd.read_csv('DiabetesData.csv')
print(df)
# Print the main statistics of each of the attributes  (i.e., mean, median, standard deviation, min, and max values)
# in a proper table
print(df.describe())
print ("shuffeld data ")
df = df.sample(frac=1)
# print the shuffled DataFrame  training a machine learning model or conducting statistical analysis
print("\nShuffled DataFrame:")
print(df)
print("====================================")
zero_count = (df['Diabetic'] == 0).sum()
one_count = (df['Diabetic'] == 1).sum()
print("negative --> Number of Zeros:", zero_count)
print(" positive -->Number of Ones:", one_count)
x=["negative", "positive"]
y=[zero_count, one_count]
plt.figure("distribution of the target class")
plt.bar(x, y, fc="lightgray", ec="black")

# =========================
print ("==========")
# to make drop  Diabetic one colum
x=df.drop('Diabetic',axis=1)
y=df['Diabetic']
# data_feature_names = ['NPG','PGL','DIA','TSF','INS','BMI','DPF','AGE','Diabetic']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7, test_size=0.3)
print("Training data - X shape:", x_train.shape)
print("Training data - y shape:", y_train.shape)
print("Test data - X shape:", x_test.shape)
print("Test data - y shape:", y_test.shape)
# Dision Tree -->
print("----Dision Tree implement M1--")
decisionModel = DecisionTreeClassifier()
# making train Data
decisionModel.fit(x_train, y_train)

y_pred = decisionModel.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#================================= M2 ==============================
print ("========== sklearn.tree.DecisionTreeClassifier¶------------------------")
x2=df.drop('Diabetic',axis=1)
y2=df['Diabetic']
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,train_size=0.5, test_size=0.5)
print("Training data - X2 shape:", x2_train.shape)
print("Training data - y2 shape:", y2_train.shape)
print("Test data - X2 shape:", x2_test.shape)
print("Test data - y2 shape:", y2_test.shape)
# # Dision Tree -->
print("----decisionModel Tree implement M2--")
decisionModel2 = DecisionTreeClassifier()
decisionModel2.fit(x2_train, y2_train)
y2_pred = decisionModel2.predict(x2_test)
accuracy2 = accuracy_score(y2_test, y2_pred)
print("Accuracy for M2--->:", accuracy2)
print("-------------- polit -------------")
plt.figure("Decision Tree M1 ",figsize=(20, 20))
plot_tree(decisionModel,filled=True, rounded=True, fontsize=5, max_depth=5)
plt.show()
plt.figure("Decision Tree M2 ",figsize=(20, 20))
plot_tree(decisionModel2,filled=True, rounded=True, fontsize=5, max_depth=5)
plt.show()