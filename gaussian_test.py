import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from naive_bayes_gaussian import naive_bayes_gaussian
from sklearn.preprocessing import OrdinalEncoder

def visualize_feature_distributions(data, feature_name):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=feature_name, hue='Age', kde=True, element='step')
    plt.title(f'Distribution of {feature_name} by Age Group')
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.legend(title='Age Group')
    plt.show()


data = pd.read_csv('abalone.csv')

# column Age from numeric 'Rings'.
# Bins: <=8 -> Young, 9-11 -> Adult, >=12 -> Old
data['Age'] = pd.cut(data['Rings'],
                     bins=[-float('inf'), 8, 11, float('inf')],
                     labels=['Young', 'Adult', 'Old'],
                     right=True)
# print(data[['Rings', 'Age']].head())

data = data.drop('Rings', axis=1)

#convert Sex to float
data['Sex'] = data['Sex'].astype('category').cat.codes
# print(data.head())


#split
X_train, X_test, y_train, y_test = train_test_split(data.drop('Age', axis=1), data['Age'], test_size=0.2, random_state=42)

#scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#fit
model = naive_bayes_gaussian()
model.fit(X_train, y_train)

#predict and calculate metrics
def calculate_accuracy(y_test, y_pred):
    return np.mean(y_test == y_pred)

y_pred = model.predict(X_test)
print("accuracy of gaussian naive bayes on abalone dataset: ", calculate_accuracy(y_test, y_pred))

#visualize all features distributions by age group
for feature in data.columns[:-1]: #exclude Age
    visualize_feature_distributions(data, feature)


