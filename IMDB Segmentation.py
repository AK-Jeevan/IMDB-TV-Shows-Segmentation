# Grouping TV shows with similar attributes using Hierarchical Clustering
# Evaluating the optimal number of clusters using Silhouette Score

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your IMDB TV shows data
# Update path or filename if needed
csv_path = r"C:\Users\akjee\Documents\AI\ML\Unsupervised Learning\IMDB.csv"
df = pd.read_csv(csv_path)
print(df.head())
df.dropna(inplace=True)

# Ensure 'Year' is string type
df['Year'] = df['Year'].astype(str)

# Extract start year from ranges like '2008–2013' or '1999-2001'
df['Year'] = df['Year'].str.extract(r'(\d{4})')  # grabs the first 4-digit number

# Convert to float
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Drop rows with invalid year
df.dropna(subset=['Year'], inplace=True)

# Encode categorical 'Type' if exists
if 'Type' in df.columns:
    df['Type'] = df['Type'].astype(str)  # ensure string
    le = LabelEncoder()
    df['Type_encoded'] = le.fit_transform(df['Type'])
    
# Select numeric features for clustering.
features = ['Rating', 'Year','Type_encoded'] if 'Type_encoded' in df.columns else ['Rating', 'Year', 'Type']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optionally reduce to 2 components for visualization
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

# Plot dendrogram using original scaled data
plt.figure(figsize=(10, 7))
linked = linkage(X_scaled, method='complete')
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean distances')
plt.show()

# Determine optimal number of clusters via silhouette score (no scipy used)
sil_scores = []
cluster_range = range(2, 11)
for n in cluster_range:
    model = AgglomerativeClustering(n_clusters=n)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(8, 4))
plt.plot(list(cluster_range), sil_scores, marker='o')
plt.xticks(list(cluster_range))
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.grid(alpha=0.3)
plt.show()

# Choose best number of clusters
best_n = cluster_range[np.argmax(sil_scores)]
print(f"Optimal number of clusters (highest silhouette): {best_n}")

# Fit final model and attach cluster labels
final_model = AgglomerativeClustering(n_clusters=best_n)
df['Cluster'] = final_model.fit_predict(X_scaled)

# Visualize clusters in PCA space
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='tab10', s=60)
plt.title('IMDB Shows - Clusters (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()

# Show cluster summaries and counts
print("\nCluster counts:")
print(df['Cluster'].value_counts().sort_index())

print("\nCluster-wise feature means:")
print(df.groupby('Cluster')[features].mean().round(3))

# Display all TV shows grouped by cluster
for cluster_id in sorted(df['Cluster'].unique()):
    print(f"\n📺 Cluster {cluster_id} TV Shows:")
    cluster_df = df[df['Cluster'] == cluster_id]
    for idx, row in cluster_df.iterrows():
        show_info = f"Title: {row['Name'] if 'Name' in row else 'N/A'}, Rating: {row['Rating']}, Year: {row['Year']}, Type: {row['Type'] if 'Type' in row else 'N/A'}"
        print(show_info)
