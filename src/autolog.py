
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

import dagshub
dagshub.init(repo_owner='devutkarsh047', repo_name='YT-MLOPS-Experiments-with-MLFlow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/devutkarsh047/YT-MLOPS-Experiments-with-MLFlow.mlflow")

# load wine data
wine = load_wine()
X = wine.data
y = wine.target

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=42)

# define the model
# max_depth = 10
# n_estimators = 5
max_depth = 8
n_estimators = 5

# Mention your experiment below
# mlflow.set_experiment('YT-MLOPS-Exp1')
mlflow.autolog()
mlflow.set_experiment('YT-MLOPS-Exp3')


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,
                                n_estimators=n_estimators,
                                random_state=42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    # creating a confusion matrix plot
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt = 'd',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    #save plot
    plt.savefig("Confusion_matrix.png")

    # Log artifacts using mlflow
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author":"Dev","Project":"Wine Classification"})

    print(accuracy)
