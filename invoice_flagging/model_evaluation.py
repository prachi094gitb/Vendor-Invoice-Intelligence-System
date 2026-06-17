from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score

def train_random_forest(X_train, y_train):
    rf= RandomForestClassifier(
        random_state= 42,
        n_jobs= -1
    )

    param_grid= {
    "n_estimators" : [100, 200, 300],
    "max_depth" : [None, 4, 5, 6],
    "min_samples_split" : [2, 3, 5],
    "min_samples_leaf" : [1, 2, 5],
    "criterion" : ['gini', 'entropy']
    }

    scorer= make_scorer(f1_score)

    randomized_search = RandomizedSearchCV(
    estimator= rf,
    param_distributions=param_grid,
    scoring= scorer,
    n_iter=20,
    cv=3,
    random_state=42,
    n_jobs=-1)

    randomized_search.fit(X_train, y_train)
    return randomized_search

def evaluate_model(model, X_test, y_test, model_name):
    y_pred= model.predict(X_test)

    accuracy= accuracy_score(y_test, y_pred)
    report= classification_report(y_test, y_pred)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)


