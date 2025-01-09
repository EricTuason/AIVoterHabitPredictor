import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, explained_variance_score, mean_squared_error, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class Nonvoters:
    # Define custom labels
    labels = ['Rarely/Never', 'Sporadic', 'Always']

    # Initialize the model
    def __init__(self) -> None:
        # Load data from CSV
        df = pd.read_csv('nonvoters_dataset.csv')

        # Drop RespId column
        df = df.drop(['RespId'], axis=1)

        # Define mapping for educ variable
        educ_mapping = {
            'High school or less': 0,
            'Some college': 1,
            'College': 2
        }

        # Define mapping for income_cat variable
        income_cat_mapping = {
            'Less than $40k': 0,
            '$40-75k': 1,
            '$75-125k': 2,
            '$125k or more': 3
        }

        # Define mapping for voter_category variable
        voter_category_mapping = {
            'rarely/never': 0,
            'sporadic': 1,
            'always': 2
        }

        # Map educ, income_cat, and voter_category variables to integer values
        df['educ'] = df['educ'].map(educ_mapping)
        df['income_cat'] = df['income_cat'].map(income_cat_mapping)
        df['voter_category'] = df['voter_category'].map(voter_category_mapping)

        # Perform one-hot encoding for race and gender variables
        df = pd.get_dummies(df, columns=['race', 'gender'])

        # Split into X and y training and test sets with an 80:20 ratio
        X = df.drop('voter_category', axis=1)
        y = df['voter_category']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # Find the best model and its parameters
    def find_model(self):
        # Try many different hidden layer sizes
        param_grid = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create an instance of the DecisionTreeClassifier with adjusted class weights
        dt = DecisionTreeClassifier(class_weight={0: 2, 1: 1.86, 2: 2})

        # Create an instance of GridSearchCV with the DecisionTreeClassifier model and the parameter grid
        grid_search = GridSearchCV(dt, param_grid, cv=10)

        # Fit the GridSearchCV object on the training data
        grid_search.fit(self.X_train, self.y_train)

        # Evaluate the performance of the best model on the testing data
        return grid_search.best_params_, grid_search.best_estimator_

    # Find the accuracy of the chosen model
    def find_predictions(self, model):
        self.y_pred = model.predict(self.X_test)
        return round(accuracy_score(self.y_test, self.y_pred), 4)

    # Find and save a visual representation of the model in the form of a confusion matrix
    def save_model(self):
        # Generate the confusion matrix plot
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels)
        disp.plot()
        plt.xlabel('Predicted label', fontweight='bold')
        plt.ylabel('True label', fontweight='bold')

        # Save the plot as an image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Encode the buffer as base64
        return base64.b64encode(buffer.getvalue()).decode()
    
    # Calculate and return the chosen model's statistics:
    # Accuracy, variance, mse, precision, and recall
    def find_stats(self):
        # Calculate the accuracy of the model
        accuracy = accuracy_score(self.y_test, self.y_pred)

        # Calculate the variance of the model
        variance = explained_variance_score(self.y_test, self.y_pred)

        # Calculate the mean squared error of the model
        mse = mean_squared_error(self.y_test, self.y_pred)

        # Calculate the precision of the model
        precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)

        # Calculate the recall of the model
        recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)

        return round(accuracy, 4), round(variance, 4), round(mse, 4), round(precision, 4), round(recall, 4)
