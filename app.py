import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Function to load the data with error handling for encoding
def load_data():
    data_path = '/content/drive/MyDrive/PS4_GamesSales.csv'  # Adjust this path as needed
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='ISO-8859-1')
    return df

# Function to plot actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred, title):
    fig = px.scatter(x=y_test, y=y_pred, title=title, labels={'x': 'Actual Values', 'y': 'Predicted Values'})
    fig.add_trace(px.line(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)]).data[0])
    return fig

# Function to plot residuals
def plot_residuals(y_test, y_pred, title):
    residuals = y_test - y_pred
    fig = px.histogram(x=residuals, nbins=50, title=title, labels={'x': 'Residuals', 'y': 'Frequency'})
    return fig

# Load and preprocess the data
df = load_data()
# Assuming 'Global' is your target variable and you want to drop 'Game' column or any identifier column
X = df.drop(['Global', 'Game'], axis=1)  # Adjust these column names as needed
y = df['Global']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['number']).columns

# Preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Train models and make predictions
models = {
    "KNN": Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', KNeighborsRegressor(n_neighbors=5))]),
    "SVM": Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', SVR(kernel='linear'))]),
    "Random Forest": Pipeline(steps=[('preprocessor', preprocessor),
                                     ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
}

predictions = {}
for model_name, model_pipeline in models.items():
    model_pipeline.fit(X_train, y_train)
    predictions[model_name] = model_pipeline.predict(X_test)

# Evaluate models
evaluation_metrics = {}
for model_name, y_pred in predictions.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evaluation_metrics[model_name] = {"MSE": mse, "R2": r2}

# Streamlit app
st.title("PS4 Game Sales Analysis")

# Main streamlit app with tabs
def main():
    st.title("PS4 Game Sales Analysis")

    tabs = ["Model Performance", "Scatter Plot", "Bar Chart", "Correlation Matrix", "Game Genre Distribution", "Histogram of Year"]
    choice = st.radio("Select a Tab", tabs)

    if choice == "Model Performance":
        st.header("Model Performance")
        for model_name, metrics in evaluation_metrics.items():
            st.subheader(f"{model_name} Model")
            st.write(f"Mean Squared Error (MSE): {metrics['MSE']}")
            st.write(f"R-squared (R2): {metrics['R2']}")

            # Plot actual vs predicted values
            st.subheader(f"{model_name}: Actual vs Predicted")
            fig1 = plot_actual_vs_predicted(y_test, predictions[model_name], f"{model_name}: Actual vs Predicted")
            st.plotly_chart(fig1)

            # Plot residuals
            st.subheader(f"{model_name}: Residuals")
            fig2 = plot_residuals(y_test, predictions[model_name], f"{model_name}: Residuals")
            st.plotly_chart(fig2)

    elif choice == "Scatter Plot":
        st.header("Scatter Plot: Global Sales vs. Japan Sales")
        fig3 = px.scatter(df, x='Japan', y='Global', title='Global Sales vs. Japan Sales', labels={'Japan': 'Japan Sales (in millions)', 'Global': 'Global Sales (in millions)'})
        st.plotly_chart(fig3)

    elif choice == "Bar Chart":
        st.header("Bar Chart: Game Genre Distribution")
        if 'Genre' in df.columns:
            genre_counts = df['Genre'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            fig4 = px.bar(genre_counts, x='Genre', y='Count', title='Game Genre Distribution')
            st.plotly_chart(fig4)
        else:
            st.write("Genre column not found in the dataset.")

    elif choice == "Correlation Matrix":
        st.header("Correlation Matrix")
        numeric_cols = df.select_dtypes(include=['number']).columns
        df_numeric = df[numeric_cols]

        corr_matrix = df_numeric.corr()
        fig5 = px.imshow(corr_matrix, color_continuous_scale='RdBu', labels=dict(color='Correlation', x='Feature', y='Feature'))
        st.plotly_chart(fig5)

    elif choice == "Game Genre Distribution":
        st.header("Game Genre Distribution")
        if 'Genre' in df.columns:
            genre_counts = df['Genre'].value_counts()
            fig6 = px.pie(values=genre_counts, names=genre_counts.index, title='Game Genre Distribution')
            st.plotly_chart(fig6)
        else:
            st.write("Genre column not found in the dataset.")

    elif choice == "Histogram of Year":
        st.header("Histogram of Year")
        fig7 = px.histogram(df, x='Year', title='Histogram of Year', labels={'Year': 'Year', 'count': 'Frequency'})
        st.plotly_chart(fig7)

if _name_ == '_main_':
    main()