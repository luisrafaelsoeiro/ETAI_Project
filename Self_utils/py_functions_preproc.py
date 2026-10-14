# BASIC IMPORTS
import pandas as pd
from datetime import datetime
import numpy as np
# VISUALIZATION
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
# MODELLING
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor
# GEOLOCATION




def plot_missing_data_heatmap(df, width=1200, height=700):
    """
    Plots a heatmap of missing data in a DataFrame using Plotly.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to visualize.
        width (int): Width of the plot.
        height (int): Height of the plot.
    """
    # MISSING VALUES DATAFRAME
    missing = df.isnull().astype(int)

    # HEATMAP
    fig = go.Figure(data=go.Heatmap(
        z=missing.values,
        x=missing.columns,
        y=np.arange(len(missing)),
        colorscale=[[0, '#040317'], [1, '#edd9a8']],
        showscale=False
    ))

    # LAYOUT
    fig.update_layout(
        title=dict(
            text="Missing Data Heatmap",
            x=0.5,
            font=dict(size=24)
        ),
        width=width,
        height=height,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(tickangle=45, showgrid=False)
    )

    fig.show()


def plot_feature_histogram(df, col, width=1200, height=700):
    """
    Plots a histogram (bar chart) of counts of unique values in the specified column.

    Parameters:
    - df: pandas DataFrame
    - col: str, column name to plot counts of
    - width: int, plot width (default 1200)
    - height: int, plot height (default 700)
    """
    value_counts = df[col].value_counts().sort_index()

    fig = go.Figure(data=go.Bar(
        x=value_counts.index,
        y=value_counts.values,
        marker_color='orange'
    ))

    fig.update_layout(
        title=dict(
            text=f"Distribution of {col}",
            x=0.5,
            font=dict(size=24)
        ),
        xaxis_title=col,
        yaxis_title='Count',
        width=width,
        height=height
    )

    fig.show()





def plot_confusion_matrix(df, col1, col2):

    freq_matrix = pd.crosstab(df[col1], df[col2])

    fig = go.Figure(data=go.Heatmap(
        z=freq_matrix.values,
        x=freq_matrix.columns,
        y=freq_matrix.index,
        colorscale='Blues',
        colorbar=dict(title='Count'),
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(
                text=f"{col1} vs {col2}",
                x=0.5,
                font=dict(size=24)
                ),
        xaxis_title=col2,
        yaxis_title=col1,
        width=1200,
        height=700
    )

    fig.show()



def spendings_correlation(df):
    per_year_cols = [col for col in df.columns if col.endswith('_per_year')]
    corr = df[per_year_cols].corr()

    # MASK UPPER TRIANGLE
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)
    text = corr_masked.round(2).astype(str).values
    text[text == 'nan'] = ''

    # HEATMAP
    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=corr.columns,
        y=corr.index,
        text=text,
        texttemplate="%{text}",
        colorscale='Cividis_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation'),
        hoverongaps=False
    ))

    # LAYOUT
    fig.update_layout(
        title=dict(
                text=f"Correlation matrix of Spendings features",
                x=0.5,
                font=dict(size=24)
                ),
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange='reversed'),
        width= 1400,
        height= 900
    )

    fig.show()


def knn_imputation(df, missing_col, corr_col):
    # Select the relevant columns
    knn_data = df[[corr_col, missing_col]]

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    knn_data_scaled = scaler.fit_transform(knn_data)

    # Initialize and apply KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    imputed_scaled_array = imputer.fit_transform(knn_data_scaled)

    # Inverse transform to bring values back to original scale
    imputed_array = scaler.inverse_transform(imputed_scaled_array)

    # Assign the imputed column to the DataFrame
    df[missing_col+'_imputed'] = imputed_array[:, 1]

    # Identify which values were imputed
    imputed_mask = df[missing_col].isnull()

    # Scatter plot for original (non-missing) values
    original_points = go.Scatter(
        x=df.loc[~imputed_mask, corr_col],
        y=df.loc[~imputed_mask, missing_col+'_imputed'],
        mode='markers',
        marker=dict(color='blue', size=6, opacity=0.7),
        name='Original Values'
    )

    # Scatter plot for imputed values
    imputed_points = go.Scatter(
        x=df.loc[imputed_mask, corr_col],
        y=df.loc[imputed_mask, missing_col+'_imputed'],
        mode='markers',
        marker=dict(color='red', size=7, line=dict(color='black', width=1), symbol='circle-open'),
        name='Imputed Values'
    )

    # Layout
    layout = go.Layout(
        title=f'KNN Imputation: {missing_col} vs {corr_col}',
        xaxis=dict(title=corr_col),
        yaxis=dict(title=missing_col),
        template='plotly_white'
    )

    # Combine and show
    fig = go.Figure(data=[original_points, imputed_points], layout=layout)
    fig.show()

    # DATAFRAME CONFIGURATION
    df.drop(columns=[missing_col], inplace=True)
    df[missing_col] = df[missing_col+'_imputed']
    df.drop(columns=[missing_col+'_imputed'], inplace=True)


def vegetables_missing_values(df):
    # Drop rows where target is missing for training
    df_model = df.dropna(subset=['spend_vegetables_per_year'])

    # Separate features and target
    X = df_model.drop(columns=['spend_vegetables_per_year'])
    y = df_model['spend_vegetables_per_year']

    # Handle categorical features
    X = pd.get_dummies(X)

    # Fill missing values with median
    X = X.fillna(X.median())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(random_state=42, n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # 1. Scatter plot: actual vs predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual Spend on Vegetables")
    plt.ylabel("Predicted Spend on Vegetables")
    plt.title("Actual vs Predicted Spend (Vegetables)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.grid(True)
    plt.show()

    # 2. Residuals plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=False)
    plt.title("Residuals Distribution")
    plt.xlabel("Prediction Error")
    plt.grid(True)
    plt.show()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R² Score: {r2:.2f}")

    # Prepare the full dataset
    X_full = df.drop(columns=['spend_vegetables_per_year'])
    X_full = pd.get_dummies(X_full)
    X_full = X_full.fillna(X_full.median())

    # Predict for missing values only
    missing_mask = df['spend_vegetables_per_year'].isna()
    df.loc[missing_mask, 'spend_vegetables_per_year'] = model.predict(X_full.loc[missing_mask])

    # Drop the temporary column if it exists
    df.drop(columns=['spend_vegetables_per_year_imputed'], errors='ignore', inplace=True)

    # For visualization, recreate the mask
    original_mask = ~missing_mask
    imputed_mask = missing_mask

    # Separate imputed vs original values for plotting
    original_values = df.loc[~missing_mask, 'spend_vegetables_per_year']
    imputed_values = df.loc[missing_mask, 'spend_vegetables_per_year']

    # Plot KDE distributions
    plt.figure(figsize=(10,6))
    sns.kdeplot(original_values, label='Original Values', fill=True)
    sns.kdeplot(imputed_values, label='Imputed Values', fill=True, linestyle='--')

    plt.title("Distribution of Original vs Imputed 'spend_vegetables_per_year'")
    plt.xlabel("Spend on Vegetables Per Year")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_promotion_by_degree(df):
    """
    Plots the distribution of 'pct_prod_in_promotion' for Degree=1 and Degree=0.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with columns 'degree' and 'pct_prod_in_promotion'.
    """
    plt.figure(figsize=(8, 4))
    sns.kdeplot(df.loc[df['degree'] == 1, 'pct_prod_in_promotion'], label='Degree=1', fill=True)
    sns.kdeplot(df.loc[df['degree'] == 0, 'pct_prod_in_promotion'], label='Degree=0', fill=True)
    plt.title('Distribution of pct_prod_in_promotion by Degree')
    plt.xlabel('pct_prod_in_promotion')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()


def histogram_scatter_matrix(df):
    features = [
    'total_spend_per_year',
    'spend_vegetables_per_year',
    'lifetime_spend_videogames',
    'spend_fish_per_year'
    ]

    # SCALE DATA
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_data, columns=features)

    n = len(features)
    fig = make_subplots(
        rows=n, cols=n,
        shared_xaxes=False, 
        shared_yaxes=False,
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    for i in range(n):
        for j in range(n):
            x_feat = features[j]
            y_feat = features[i]

            if i == j:
                # HISTOGRAM
                fig.add_trace(
                    go.Histogram(
                        x=scaled_df[x_feat],
                        nbinsx=200,
                        showlegend=False,
                        marker=dict(color='lightblue')
                    ),
                    row=i+1, col=j+1
                )
            else:
                # SCATTER PLOT
                fig.add_trace(
                    go.Scattergl(
                        x=scaled_df[x_feat],
                        y=scaled_df[y_feat],
                        mode='markers',
                        marker=dict(size=2, opacity=0.6, color='darkblue'),
                        showlegend=False
                    ),
                    row=i+1, col=j+1
                )

    # LAYOUT
    fig.update_layout(
        height=1000,
        width=1800,
        title='Matrix of Histograms and Scatter Plots',
        hovermode='closest'
    )

    # AXIS LABELS
    for i in range(n):
        for j in range(n):
            show_x_label = (i == n - 1)
            show_y_label = (j == 0)

            fig.update_xaxes(
                showticklabels=show_x_label,
                title_text=features[j] if show_x_label else '',
                row=i+1, col=j+1
            )
            fig.update_yaxes(
                showticklabels=show_y_label,
                title_text=features[i] if show_y_label else '',
                row=i+1, col=j+1
            )

    fig.show()


def spend_cat_over_age(df):
    
    spending_cols = [
        'spend_groceries_per_year', 'spend_vegetables_per_year',
        'spend_nonalcohol_drinks_per_year', 'spend_meat_per_year',
        'spend_hygiene_per_year', 'spend_petfood_per_year',
        'spend_electronics_per_year', 'spend_videogames_per_year',
        'total_spend_per_year', 'spend_fish_per_year',
        'spend_alcohol_drinks_per_year', 'pct_prod_in_promotion',
    ]

    # Drop duplicates in spending columns list
    spending_cols = list(dict.fromkeys(spending_cols))

    # SCALE spending data across individuals
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[spending_cols] = scaler.fit_transform(df[spending_cols])

    # AGE BINS
    df_scaled["age_bin"] = pd.cut(df_scaled["age"], bins=range(22, int(df["age"].max()) + 1), right=True)
    df_scaled["age_bin_label"] = df_scaled["age_bin"].apply(lambda x: int(x.right))

    # VALUES (scaled spending medians per age group)
    grouped = df_scaled.groupby("age_bin_label")[spending_cols].median().T

    # PLOT
    fig = go.Figure(data=go.Heatmap(
        z=grouped.values,
        x=grouped.columns.astype(str),
        y=grouped.index,
        colorscale='YlGnBu',
        zmin=0, zmax=1,
        xgap=1, ygap=1
    ))

    fig.update_layout(
        title="Median Yearly Spending per Category Over Age",
        xaxis_title="Customer Age",
        yaxis_title="Spending Category",
        width=1500,
        height=800
    )

    fig.show()

    # Clean up temporary columns
    df_scaled.drop(columns=['age_bin', 'age_bin_label'], inplace=True)