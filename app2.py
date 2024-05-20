import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load the dataset
def load_data():
    try:
        data = pd.read_csv('Churn_Modelling.csv')
        st.write("Data loaded successfully")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Step 3: Pre-processing functions
def preprocess_data(data):
    try:
        # Perform one-hot encoding for categorical columns
        data = pd.get_dummies(data)

        # Exclude non-numeric columns from preprocessing
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

        # Calculate quartiles and IQR
        Q1 = data[numeric_columns].quantile(0.25)
        Q3 = data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1

        # Handle outliers
        for col in numeric_columns:
            Q95 = data[col].quantile(0.95)
            Q05 = data[col].quantile(0.05)
            median = data[col].quantile(0.50)
            data.loc[(data[col] > Q95), col] = median
            data.loc[(data[col] < Q05), col] = median

        st.write("Data preprocessed successfully")
        return data
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

# Step 5: Model training and evaluation
def train_and_evaluate_model(X, y):
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train models
        # Neural Network
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        mlp_accuracy = accuracy_score(y_test, mlp.predict(X_test))

        # KNN
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_accuracy = accuracy_score(y_test, knn.predict(X_test))

        # Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        dt_accuracy = accuracy_score(y_test, dt.predict(X_test))

        # Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_accuracy = accuracy_score(y_test, rf.predict(X_test))

        st.write("Model training and evaluation completed successfully")
        return mlp_accuracy, knn_accuracy, dt_accuracy, rf_accuracy
    except Exception as e:
        st.error(f"Error training and evaluating models: {e}")
        return None, None, None, None

# Step 6: Streamlit app layout
def main():
    st.title('Churn Prediction Web App')

    # Step 2: Load the data
    data = load_data()
    if data is None:
        return

    # Step 3: Preprocess the data
    preprocessed_data = preprocess_data(data)
    if preprocessed_data is None:
        return

    # Step 5: Train and evaluate models
    X = preprocessed_data.drop(columns=['Exited'])
    y = preprocessed_data['Exited']
    mlp_acc, knn_acc, dt_acc, rf_acc = train_and_evaluate_model(X, y)
    if None in (mlp_acc, knn_acc, dt_acc, rf_acc):
        return

    # Step 6: Display results
    st.subheader('Model Performance')
    st.write('Accuracy of MLP Classifier:', mlp_acc)
    st.write('Accuracy of KNN Classifier:', knn_acc)
    st.write('Accuracy of Decision Tree Classifier:', dt_acc)
    st.write('Accuracy of Random Forest Classifier:', rf_acc)

if __name__ == '__main__':
    main()
