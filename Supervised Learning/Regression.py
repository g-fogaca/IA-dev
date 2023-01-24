#%%
# importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

# importando dataset
diabetes_df = pd.read_csv("diabetes_clean.csv")

# separando variáveis independentes e variável dependente
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

# Escolhendo uma variável
X_bmi = diabetes_df["bmi"].values

# Reshape para que os arrays sejam 2d
X_bmi = X_bmi.reshape(-1,1)
y = y.reshape(-1,1)

# Visualizando variáveis
plt.scatter(X_bmi, y)
plt.xlabel("Body Mass Index")
plt.ylabel("Blood Glucose (mg/dl)")
plt.show()

# Rodando Regressão
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

# Visualizando Regressão
plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions, c="r")
plt.xlabel("Body Mass Index")
plt.ylabel("Blood Glucose (mg/dl)")
plt.show()

#%%
# importando dataset
sales_df = pd.read_csv("advertising_and_sales_clean.csv")
sales_df = sales_df.drop("influencer", axis=1)

# Create X from the radio column's values
X = sales_df["radio"].values

# Create y from the sales column's values
y = sales_df["sales"].values

# Reshape X
X = X.reshape(-1,1)

# Check the shape of the features and targets
print(X.shape, y.shape)

#%%
# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])

#%%
# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()

#%%
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=42)

reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

reg_all.score(X_test, y_test)
mean_squared_error(y_test, y_pred, squared = False)

#%%
# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2],
                                                  y_test[:2]))

#%%
# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))

#%%
# Create a KFold object
kf = KFold(n_splits = 6, shuffle = True, random_state = 5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv = kf)

# Print scores
print(cv_scores)

#%%

cv_results = cv_scores
# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))

#%%
scores = []
for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))
print(scores)

scores = []
for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_pred = ridge.predict(X_test)
    scores.append(lasso.score(X_test, y_test))
print(scores)

#%%
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values
names = diabetes_df.drop("glucose", axis=1).columns

lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X,y).coef_

plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()
