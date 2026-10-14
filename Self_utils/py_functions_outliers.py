# BASIC IMPORTS
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# VISUALIZATION
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interact
from upsetplot import UpSet
from upsetplot import from_indicators

# MODELLING
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from minisom import MiniSom
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import itertools
from scipy.stats import chi2
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE


def plot_outlier_detection(df, spend_col, multiplier=10):
    """
    Plots box plots and histograms side-by-side for each feature in spend_col.
    Red vertical lines on histograms show IQR-based outlier boundaries.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - spend_col: list of column names (continuous features) to plot
    """
    for col in spend_col:
        data = df[col].dropna()
        
        # IQR AND OUTLIER BOUNDARY
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + multiplier * iqr
        
        # SUBPLOTS
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.3, 0.7],
            subplot_titles=(f"Box Plot - {col}", f"Histogram - {col}")
        )

        # BOX PLOT
        fig.add_trace(
            go.Box(y=data, name=col, boxpoints="outliers", marker_color='blue'),
            row=1, col=1
        )

        # HISTOGRAM
        fig.add_trace(
            go.Histogram(x=data, nbinsx=200, marker_color='lightblue'),
            row=1, col=2
        )

        # OUTLIER BOUNDARY
        fig.add_vline(x=upper_bound, line=dict(color="red", width=2, dash="dash"), row=1, col=2)

        # LAYOUT
        fig.update_layout(
            title_text=f"Outlier Detection for {col}",
            showlegend=False,
            height=600,
            width=1500
        )
        fig.show()


def slice_outliers(df, slice=False):
    """
    Removes outliers using the IQR method from all numeric columns.
    
    Parameters:
    - df: DataFrame
    - slice (bool): Whether to return the dataframe without outliers or not

    Returns:
    - (DataFrame without outliers, DataFrame of all outliers)
    """
    numeric_columns = df.select_dtypes(include=np.number).columns
    df_clean = df.copy()
    outliers_all = pd.DataFrame()

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 10 * IQR
        upper_bound = Q3 + 10 * IQR

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers_all = pd.concat([outliers_all, outliers])
        df_clean = df_clean[~((df_clean[column] < lower_bound) | (df_clean[column] > upper_bound))]

    outliers_all = outliers_all.drop_duplicates()

    if slice:
        return df_clean, outliers_all
    return df, outliers_all

def plot_tsne_dbscan(X_scaled, df2, perplexity=30, random_state=42):
    """
    Perform t-SNE dimensionality reduction and plot DBSCAN clustering with outliers.

    Parameters:
    - X_scaled: np.ndarray or pd.DataFrame, scaled feature matrix
    - df2: pd.DataFrame, must contain 'dbscan_cluster' and 'db_scan_outlier' columns
    - perplexity: int, t-SNE perplexity parameter
    - random_state: int, random seed for reproducibility
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)
    y_sample = df2['dbscan_cluster']

    plt.figure(figsize=(18, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y_sample,
        palette='Set1',
        style=df2['db_scan_outlier'].map({True: 'X', False: 'o'}),
        legend='full'
    )
    plt.title("DBSCAN Clustering & Outliers (t-SNE reduced)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


def plot_tsne_kmeans_outliers(X_scaled, df2, perplexity=30, random_state=42):
    """
    Perform t-SNE dimensionality reduction and plot KMeans clustering with outliers.

    Parameters:
    - X_scaled: np.ndarray or pd.DataFrame, scaled feature matrix
    - df2: pd.DataFrame, must contain 'kmeans_outlier' column
    - perplexity: int, t-SNE perplexity parameter
    - random_state: int, random seed for reproducibility
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)

    # Add a column for plotting
    df2['outlier_flag'] = df2['kmeans_outlier'].map({True: 'Outlier', False: 'Inlier'})

    plt.figure(figsize=(18, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df2['outlier_flag'],
        palette={'Inlier': 'blue', 'Outlier': 'red'},
        style=df2['outlier_flag'].map({'Outlier': 'X', 'Inlier': 'o'}),
        legend='full'
    )
    plt.title("KMeans Clustering – Outliers in Red (t-SNE Reduced)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


def plot_kmeans_cluster_bar(df2):
    """
    Create a stacked bar chart showing the number of inliers and outliers per KMeans cluster.

    Parameters:
    - df2: pd.DataFrame, must contain 'kmeans_cluster' and 'outlier_flag' columns
    """
    # Group data
    cluster_counts = df2.groupby(['kmeans_cluster', 'outlier_flag']).size().reset_index(name='count')

    # Split into two DataFrames
    inliers = cluster_counts[cluster_counts['outlier_flag'] == 'Inlier']
    outliers = cluster_counts[cluster_counts['outlier_flag'] == 'Outlier']

    # Ensure all clusters are represented
    all_clusters = sorted(df2['kmeans_cluster'].unique())

    inlier_counts = pd.Series(inliers.set_index('kmeans_cluster')['count'], index=all_clusters).fillna(0)
    outlier_counts = pd.Series(outliers.set_index('kmeans_cluster')['count'], index=all_clusters).fillna(0)

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=all_clusters,
        y=inlier_counts,
        name='Inlier',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=all_clusters,
        y=outlier_counts,
        name='Outlier',
        marker_color='red'
    ))

    # Update layout
    fig.update_layout(
        barmode='stack',
        title='Number of Observations per KMeans Cluster',
        xaxis_title='KMeans Cluster',
        yaxis_title='Number of Observations',
        xaxis=dict(type='category'),
        legend_title='Observation Type',
        template='plotly_white',
        height=500,
        width=1700
    )

    fig.show()


def dbscan_grid_search_with_silhouette(X_scaled, eps_range=(1, 4), eps_steps=5, min_samples_values=[4, 5, 7, 9]):
    """
    Performs grid search on DBSCAN parameters using silhouette score as evaluation metric.

    Parameters:
    - X_scaled (ndarray or DataFrame): Scaled input features for clustering.
    - eps_range (tuple): Range (start, stop) for eps values in linspace.
    - eps_steps (int): Number of steps/values for eps.
    - min_samples_values (list): List of min_samples values to try.

    Returns:
    - results_df (DataFrame): DataFrame of all grid search results.
    - best (DataFrame): Top 5 parameter sets sorted by silhouette score.
    """
    results = []
    eps_values = np.linspace(eps_range[0], eps_range[1], eps_steps)

    for eps, min_samples in itertools.product(eps_values, min_samples_values):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        if n_clusters > 1 and n_noise < len(labels):
            sil_score = silhouette_score(X_scaled, labels, metric='euclidean')
        else:
            sil_score = np.nan

        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': sil_score
        })

    results_df = pd.DataFrame(results)
    best = results_df.sort_values(by='silhouette_score', ascending=False)
    return results_df, best.head(5)


def plot_elbow_silhouette(X, max_k=100):
    '''
    Function to plot both elbow method and silhouette score on the same plot.
    
    Arguments: 
    - X(pd.DataFrame): DataFrame with variables to plot.
    - max_k(int): Maximum number of clusters allowed
    
    Returns:
    - None, although a plot is produced.
    '''
    # Elbow and Silhouette evaluation
    def evaluate_kmeans(X=X, max_k=max_k):
        inertias = []
        silhouettes = []
        K_range = range(2, max_k+1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            labels = kmeans.labels_
            silhouette = silhouette_score(X, labels)
            silhouettes.append(silhouette)

        return inertias, silhouettes, K_range

    inertias, silhouettes, K_range = evaluate_kmeans(X)

    # ---- Combined Plot: Elbow (Inertia) and Silhouette ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia (Elbow)', color=color1)
    ax1.plot(K_range, inertias, marker='o', color=color1, label='Inertia')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create a second y-axis
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Silhouette Score', color=color2)
    ax2.plot(K_range, silhouettes, marker='o', linestyle='--', color=color2, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Titles and legend
    plt.title('Elbow Method and Silhouette Score vs. Number of Clusters')
    fig.tight_layout()
    plt.show()


def plot_tsne_outliers(X_scaled, df, outlier_col='som_outlier', perplexity=30, random_state=42):
    """
    Performs t-SNE dimensionality reduction and plots 2D scatter plot
    colored by outlier status.

    Parameters:
    - X_scaled (ndarray or DataFrame): Scaled input features.
    - df (pd.DataFrame): Original DataFrame containing the outlier flag.
    - outlier_col (str): Name of the column in `df` indicating outliers as booleans.
    - perplexity (int): t-SNE perplexity parameter.
    - random_state (int): Random state for reproducibility.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)

    df['outlier_flag'] = df[outlier_col].map({True: 'Outlier', False: 'Inlier'})

    plt.figure(figsize=(18, 8))
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=df['outlier_flag'],
        palette={'Inlier': 'blue', 'Outlier': 'red'},
        style=df['outlier_flag'].map({'Outlier': 'X', 'Inlier': 'o'}),
        legend='full'
    )
    plt.title("SOM Clustering – Outliers in Red (t-SNE Reduced)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


def plot_outlier_agreement(df):
    """
    Plots an UpSet chart showing agreement between different outlier detection methods.

    Parameters:
    df (pd.DataFrame): A DataFrame containing at least the columns 'db_scan_outlier', 'kmeans_outlier', and 'som_outlier'.
    """
    outlier_df = df[['db_scan_outlier', 'kmeans_outlier', 'som_outlier']]
    data = from_indicators(outlier_df.columns.tolist(), outlier_df)

    fig = plt.figure(figsize=(24, 14))
    upset = UpSet(data, show_counts=True)
    upset.plot(fig=fig)

    fig.suptitle("Outlier Agreement")
    plt.tight_layout()
    plt.show()