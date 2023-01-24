# importando bibliotecas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# importando dados
diabetes_df = pd.read_csv("diabetes_clean.csv")

# definindo variáveis feature e target
X = diabetes_df.drop("glucose", axis=1).values
y = diabetes_df["glucose"].values

# iniciando modelo
knn = KNeighborsClassifier(n_neighbors = 6)

# Separando em conjuntos treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                    test_size=0.3, random_state=42)

# Aplicando o modelo
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculando métricas do modelo
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
# Logistic Regression
# Importando modelo
from sklearn.linear_model import LogisticRegression

# iniciando modelo
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print(y_pred_probs[:10])

#%%
# Import roc_curve
from sklearn.metrics import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()

#%%
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))
