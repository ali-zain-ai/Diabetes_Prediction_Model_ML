import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv(r'D:\AI_Python\Dabaties Patient\diabetes.csv')
print(df.head(5))
# print("\n_____Information of Dataset_____\n")
# print(df.info())
# print("\n_____The statistical summary of dataset_____\n")
# print(df.describe())

#now we are checking that how many patientes are diabetic or not diabetic
# print(df["Outcome"].value_counts())      # 500 --> non diabetic / 268 --> diabetic


#it will tell the mean of diabetic and non diabetic patients for all columns 
# print(df.groupby('Outcome').mean())


#now we are seperating the labes and the features 
X = df.drop(columns=('Outcome'), axis=1)
# print(X)
y = df['Outcome']
# print(y)

#data standerdization

scalar = StandardScaler()
scalar.fit(X)
standerdized_data = scalar.transform(X)

X= standerdized_data
Y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2,random_state=1)

#training the model 

model = LogisticRegression(max_iter=2500)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)
# print(f"The result of prediction by giving the featues of triang data: {y_pred}\n")
accuracy = accuracy_score(y_train, y_pred)
print(f'Model accuracy = {accuracy*100:.2f}%' )
print("Classification_Report\n",classification_report(y_train, y_pred))