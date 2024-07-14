import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
from sklearn.ensemble import VotingClassifier


######## train data ##########
# data = pd.read_csv("new_data.csv")
balanced_data = pd.read_csv("new_data_Balance1.csv")
balanced_data = balanced_data.reset_index(drop=True)
y = balanced_data["fraudulent"]
x = balanced_data.drop("fraudulent", axis=1)
x = x.drop(balanced_data.columns[:7], axis=1)
x = x.drop(balanced_data.columns[10:15], axis=1)
x = x.drop("country", axis=1)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y, random_state=123)

########### Decition tree #################
def importence_DT(X_train, y_train, X_val, y_val):
    tree = DecisionTreeClassifier(class_weight={0: 1.0, 1: 3.0}, max_depth=10,
                                  random_state=42)
    tree.fit(X_train, y_train)
    plt.figure(figsize=(12, 10))
    plot_tree(tree, filled=True, class_names=True, max_depth=2, feature_names=x.columns)
    plt.show()
    print(f"Test accuracy: {roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1]):.2}")
    print(f"Test accuracy: {roc_auc_score(y_val, tree.predict_proba(X_val)[:, 1]):.2}")

    feature_names = x.columns
    importances = tree.feature_importances_
    # Create a dataframe to display the feature importances
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    print(feature_importances.head(10))
def decisionTreeModel(X_train, y_train, X_val, y_val):
    param_grid = {
        'max_depth': np.arange(1, 36, 1),
        'criterion': ['entropy', 'gini'],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [
            {0: 1.0, 1: 2.0},  # Option 1
            {0: 1.0, 1: 1.5},  # Option 2
            {0: 1.0, 1: 1.0},  # Option 3 (equal weights)
            {0: 0.5, 1: 2.0},  # Option 4
            {0: 1.0, 1: 3.0},  # Option 5

        ]
    }
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                               param_grid=param_grid,
                               refit=True,
                               cv=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(best_model)
    print(f"Test accuracy: {roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1]):.2}")
    print(f"Test accuracy: {roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1]):.2}")
    y_pred = best_model.predict(X_val)
    print(classification_report(y_val, y_pred))
def confusion_matrix_plot(y_true, y_predict):
    cm = confusion_matrix(y_true, y_predict)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = np.unique(y_true)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix - Validation',
           ylabel='True label',
           xlabel='Predicted label')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.show()
############ LinearSVC #################
def LinearSVC_PCA(X_train, y_train, X_val, y_val):
    pca = PCA(n_components=2)
    pca.fit(X_train)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    pca1 = PCA(n_components=2)
    pca1.fit(X_val)
    train_pca = pca.transform(X_train)
    train_pca = pd.DataFrame(train_pca, columns=[f'PC{i + 1}' for i in range(2)])
    val_pca = pca.transform(X_val)
    val_pca = pd.DataFrame(val_pca, columns=[f'PC{i + 1}' for i in range(2)])
    # Reset indices of X_train and y_train
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    param_grid = {
        'C': np.arange(1, 30, 1),
        'loss': ['hinge', 'squared_hinge'],
    }
    grid_search = GridSearchCV(estimator=LinearSVC(random_state=42),
                               param_grid=param_grid,
                               refit=True,
                               cv=10)
    grid_search.fit(train_pca, y_train)
    best_model = grid_search.best_estimator_
    print(best_model)
def LinearSVC_Plot(X_train,y_train):
    model = LinearSVC(C=22, random_state=42)
    selected_features = ['company_profile_len', 'title_len']  # Specify the names of the features you want to use
    X_train_selected = X_train[selected_features]
    model.fit(X_train_selected, y_train)
    feature1_values = np.linspace(X_train_selected['title_len'].min() - 1, X_train_selected['title_len'].max() + 1, 100)
    feature2_values = np.linspace(X_train_selected['company_profile_len'].min() - 1,
                                  X_train_selected['company_profile_len'].max() + 1, 100)
    xx, yy = np.meshgrid(feature1_values, feature2_values)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_train_selected['company_profile_len'][y_train == 0], X_train_selected['title_len'][y_train == 0],
                c='red', label='Class 0')
    plt.scatter(X_train_selected['company_profile_len'][y_train == 1], X_train_selected['title_len'][y_train == 1],
                c='blue', label='Class 1')
    plt.xlabel('Description Length')
    plt.ylabel('Company Profile Length')
    plt.title('Separate Fields Based on Description Length and Company Profile Length')
    plt.legend()
    plt.show()
def print_SVC_equation(model):
    # Obtain the coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    # Construct the equation of the decision boundary
    # equation = " + ".join([f"({coefficients[0, i]:.2f}) * x{i+1}" for i in range(coefficients.shape[1])])
    # equation += f" + ({intercept[0]:.2f}) = 0"
    equation = " + ".join([f"({coefficients[0, i]:.2f}) * x{i + 1}" for i in range(coefficients.shape[1])])
    equation += f" + ({intercept[0]:.2f}) = 0"

    # Split equation into multiple rows with 10 coefficients in each row
    coefficients_split = equation.split(" + ")
    equation_split = [coefficients_split[i:i + 10] for i in range(0, len(coefficients_split), 10)]
    equation_rows = [" + ".join(row) for row in equation_split]

    equation_formatted = " + \n".join(equation_rows)
    equation_formatted += " = 0"

    print(equation_formatted)
def Linear_SVC(X_train, y_train, X_val, y_val):
    param_grid = {
        'C': np.arange(1, 30, 1),
        'loss': ['hinge', 'squared_hinge'],
    }
    grid_search = GridSearchCV(estimator=LinearSVC(random_state=42),
                               param_grid=param_grid,
                               refit=True,
                               cv=10)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(best_model)
    print(f"train accuracy: {roc_auc_score(y_train, best_model._predict_proba_lr(X_train)[:, 1]):.2}")
    print(f"val accuracy: {roc_auc_score(y_val, best_model._predict_proba_lr(X_val)[:, 1]):.2}")
    print(confusion_matrix(y_true=y_train, y_pred=best_model.predict(X_train)))
################ MLP ###################
def MLP_defualt(X_train, y_train, X_val, y_val):
    model = MLPClassifier()
    model.fit(X_train, y_train)
    print(f"Test accuracy: {roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]):.2}")
    print(f"val accuracy: {roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]):.2}")
    print("Number of layers:", model.n_layers_)
    print("Number of features:", model.n_features_in_)
    print("Number of classes:", model.n_outputs_)
    print("Hidden layer sizes:", model.hidden_layer_sizes)
    print("Output activation function:", model.out_activation_)
    print(confusion_matrix(y_true=y_train, y_pred=model.predict(X_train)))
    print(confusion_matrix(y_true=y_val, y_pred=model.predict(X_val)))
def MLP_model(X_train, y_train, X_val, y_val):
    hidden_layer_sizes = []
    for i in range(1, 101, 2):
        hidden_layer_sizes.append((i,))
    param_grid = {
        'hidden_layer_sizes': (75,),
        'activation': ['relu', 'logistic', 'tanh']
        , 'alpha': np.arange(0.3, 0.7, 0.1)
        , 'max_iter': np.arange(100, 400, 100)
        , 'learning_rate_init': [0.01, 0.001, 0.001, 0.1]
    }
    grid_search = GridSearchCV(estimator=MLPClassifier(random_state=42),
                               param_grid=param_grid,
                               refit=True,
                               cv=10)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_model = MLPClassifier(hidden_layer_sizes=(75,), max_iter=200, random_state=42
                               , alpha=0.5, activation='relu', solver='adam', learning_rate_init=0.001)
    best_model.fit(X_train, y_train)

    print(best_model)
    print(f"Train accuracy: {roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1]):.2f}")
    print(f"Validation accuracy: {roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1]):.2f}")

    # Plot the loss curve
    plt.plot(best_model.loss_curve_)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    train_accs = []
    test_accs = []
    for size_ in range(1, 100, 2):
        print(f"size: {size_}")
        model = MLPClassifier(random_state=42,
                              hidden_layer_sizes=(size_, size_), max_iter=1000,
                              activation='relu',
                              learning_rate_init=0.01)
        model.fit(X_train, y_train)
        train_acc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        train_accs.append(train_acc)
        test_acc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        test_accs.append(test_acc)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 100, 2), train_accs, label='Train')
    plt.plot(range(1, 100, 2), test_accs, label='Test')
    plt.legend()
    plt.xlabel('# neurons')
    plt.title('Tuning the number of neurons in 2 layers networks ')
    plt.show()
    print(f"Test accuracy: {roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1]):.2}")
    print(f"val accuracy: {roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1]):.2}")
    print(confusion_matrix(y_true=y_train, y_pred=best_model.predict_proba(X_train)))
    print(confusion_matrix(y_true=y_val, y_pred=best_model.predict_proba(X_val)))
    confusion_matrix_plot(y_val, best_model.predict(X_val))
########### Clustering #################
def clustering(X_train, y_train, X_val, y_val):
    x_cluster = X_train[
        ['has_company_logo', 'has_questions', 'description_len', "benefits_num_words", 'requirements_num_words',
         'requirements_len', 'company_profile_len', 'title_len', 'et_Full-time', 'et_Part-time', 'et_Other']]
    iner_list = []
    dbi_list = []
    sil_list = []

    for n_clusters in range(2, 10, 1):
        kmeans = KMedoids(n_clusters=n_clusters, random_state=42)
        kmeans.fit(x_cluster)
        assignment = kmeans.predict(x_cluster)
        print(assignment)
        iner = kmeans.inertia_
        sil = silhouette_score(x_cluster, assignment)
        dbi = davies_bouldin_score(x_cluster, assignment)

        dbi_list.append(dbi)
        sil_list.append(sil)
        iner_list.append(iner)
    plt.plot(range(2, 10, 1), iner_list, marker='o')
    plt.title("Inertia")
    plt.xlabel("Number of clusters")
    plt.show()

    plt.plot(range(2, 10, 1), sil_list, marker='o')
    plt.title("Silhouette")
    plt.xlabel("Number of clusters")
    plt.show()

    plt.plot(range(2, 10, 1), dbi_list, marker='o')
    plt.title("Davies-bouldin")
    plt.xlabel("Number of clusters")
    plt.show()
####### Voting #########
def voting(X_train, X_val, y_train, y_val):
    classifier1 = MLPClassifier(hidden_layer_sizes=(75,), max_iter=200, random_state=42, alpha=0.5, activation='relu',
                                solver='adam', learning_rate_init=0.001)
    classifier2 = MLPClassifier(hidden_layer_sizes=(75, 75, 75), activation='relu', random_state=42, alpha=0.5)
    classifier3 = MLPClassifier(hidden_layer_sizes=(75, 75), activation='tanh', random_state=42, alpha=0.5)
    voting_classifier = VotingClassifier(
        estimators=[('ann1', classifier1), ('ann2', classifier2), ('ann3', classifier3)],
        voting='soft'

    )
    voting_classifier.fit(X_train, y_train)
    print(f"train accuracy: {roc_auc_score(y_train, voting_classifier.predict_proba(X_train)[:, 1]):.2}")
    print(f"val accuracy: {roc_auc_score(y_val, voting_classifier.predict_proba(X_val)[:, 1]):.2}")
########### changes on the data ##############
def data_change(X_train, X_val, y_train, y_val):
    binary = X_train.drop(['description_len','title_len','company_profile_len','requirements_len','requirements_num_words','benefits_num_words'],axis = 1)
    binary_val = X_val.drop(['description_len','title_len','company_profile_len','requirements_len','requirements_num_words','benefits_num_words'],axis = 1)
    for index, row in binary.iterrows():
        for feature in binary.columns:
            if row[feature] == 0:
                binary.at[index, feature] = -1
    for index, row in binary_val.iterrows():
        for feature in binary_val.columns:
            if row[feature] == 0:
                binary_val.at[index, feature] = -1
    best_model = MLPClassifier(hidden_layer_sizes=(75,60), max_iter=200, random_state=42
                               , alpha=0.5, activation='tanh', solver='adam', learning_rate_init=0.001)
    best_model.fit(binary,y_train)
    print(best_model)
    print(f"Train accuracy: {roc_auc_score(y_train, best_model.predict_proba(binary)[:, 1]):.2}")
    print(f"val accuracy: {roc_auc_score(y_val, best_model.predict_proba(binary_val)[:, 1]):.2}")

