import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score, classification_report, \
    roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def load_data():
    # Load the data from CSV
    df = pd.read_csv('YY_Mortality_District.csv')
    return df


def kmeans_clustering(df):
    # Extract relevant columns
    X = df[['YY_Crude_Death_Rate_Cdr_Total_Male', 'YY_Crude_Death_Rate_Cdr_Total_Female']].dropna()

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Add cluster labels to DataFrame
    df['Cluster'] = np.nan
    df.loc[X.index, 'Cluster'] = clusters

    # Plot the clusters and save the figure
    plt.figure(figsize=(10, 6))
    plt.scatter(df['YY_Crude_Death_Rate_Cdr_Total_Male'], df['YY_Crude_Death_Rate_Cdr_Total_Female'], c=df['Cluster'],
                cmap='viridis')
    plt.xlabel('Crude Death Rate Total Male')
    plt.ylabel('Crude Death Rate Total Female')
    plt.title('K-Means Clustering of Death Rates')
    plt.colorbar(label='Cluster')
    plt.savefig('kmeans_clustering.png')
    plt.close()


def build_and_evaluate_ann(df):
    # Prepare data
    X = df[['YY_Infant_Mortality_Rate_Imr_Urban_Person']].dropna()
    y = (X > X.median()).astype(int)  # Example binary classification

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build ANN
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")


def build_and_evaluate_classification_model(df):
    # Prepare data
    X = df[['YY_Crude_Death_Rate_Cdr_Total_Person']].dropna()
    y = (X > X.median()).astype(int)  # Example binary classification

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build ANN
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred)}")


def plot_death_rates(df):
    # Plot death rates by state and save the figure
    plt.figure(figsize=(12, 8))
    df.groupby('State_Name')['YY_Crude_Death_Rate_Cdr_Total_Person'].mean().plot(kind='bar')
    plt.xlabel('State')
    plt.ylabel('Average Crude Death Rate')
    plt.title('Average Crude Death Rate by State')
    plt.xticks(rotation=90)
    plt.savefig('death_rates_by_state.png')
    plt.close()


def main():
    df = load_data()
    kmeans_clustering(df)
    build_and_evaluate_ann(df)
    build_and_evaluate_classification_model(df)
    plot_death_rates(df)


if __name__ == "__main__":
    main()
