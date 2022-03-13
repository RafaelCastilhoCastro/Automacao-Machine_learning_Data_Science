import pandas
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Imports DB
table1 = pandas.read_csv("Python/Automation - Machine Learning & Data Science/advertising.csv")

# Create & display a correlation chart
seaborn.heatmap(table1.corr(), cmap="Wistia", annot=True)
plt.show()

# Spit DB train & test parts
y = table1["Vendas"]
x = table1[["TV", "Radio", "Jornal"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create AI modules
model_linearRegression = LinearRegression()
model_decisionTree = RandomForestRegressor()

# Train AI modules
model_linearRegression.fit(x_train, y_train)
model_decisionTree.fit(x_train, y_train)

# Test AI modules & predict results
predict_linearRegression = model_linearRegression.predict(x_test)
predict_decisionTree = model_decisionTree.predict(x_test)

# Display R² (comparison between predictions and y_test)
print(metrics.r2_score(y_test, predict_linearRegression))
print(metrics.r2_score(y_test, predict_decisionTree))

'''
Create and display a chart showing the comparison between the y_test values
and the predictions of both AI models
'''
table2 = pandas.DataFrame()
table2["y_test"] = y_test
table2["Linear Regression Prediction"] = predict_linearRegression
table2["Decision Tree Prediction"] = predict_decisionTree
plt.figure(figsize=(15, 6))
seaborn.lineplot(data=table2)
plt.show()

# Importing new data to perform new predictions
table3 = pandas.read_csv("Python/Automation - Machine Learning & Data Science/novos.csv")

# Use AI to make a new prediction
new_prediction = model_decisionTree.predict(table3)
print(new_prediction)
