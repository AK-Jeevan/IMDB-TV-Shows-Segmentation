# 📺 IMDB TV Show Segmentation using Hierarchical Clustering

This project applies unsupervised learning to segment TV shows from IMDB based on attributes like rating, release year, and type. It uses **Hierarchical Clustering** and evaluates cluster quality using the **Silhouette Score**, with visualizations powered by PCA and dendrograms.

## 🔍 Objective

To discover natural groupings among TV shows using hierarchical clustering and visualize them in reduced dimensions for interpretability.

## 🧰 Technologies Used

- **Python**
- **Pandas** & **NumPy** – data manipulation
- **scikit-learn** – preprocessing, clustering, PCA, silhouette analysis
- **SciPy** – dendrogram generation
- **Matplotlib** & **Seaborn** – visualization

## 📁 Dataset

The dataset includes:
- `Title` – name of the TV show
- `Rating` – IMDB rating
- `Year` – release year (cleaned from ranges like `'2008–2013'`)
- `Type` – format (e.g., Series, Mini-Series)

## 🧪 Methodology

1. **Data Cleaning**
   - Removed missing values
   - Extracted start year from ranges using regex
   - Converted year to float
   - Encoded categorical `Type` feature

2. **Feature Selection**
   - Selected `Rating`, `Year`, and encoded `Type` for clustering

3. **Preprocessing**
   - Standardized features using `StandardScaler`
   - Reduced dimensions to 2D using `PCA` for visualization

4. **Clustering**
   - Applied **Agglomerative Clustering** with varying cluster counts
   - Evaluated cluster quality using **Silhouette Score**
   - Selected optimal number of clusters based on highest score

5. **Visualization**
   - Dendrogram to show hierarchical structure
   - PCA scatterplot to visualize clusters
   - Cluster summaries and grouped TV show listings

## 📊 Results

- Optimal number of clusters determined via silhouette analysis
- Shows grouped meaningfully based on rating, year, and type
- Visual insights into cluster distribution and feature averages

## 🚀 How to Run

Clone the repository:

   git clone https://github.com/your-username/imdb-clustering.git

Install dependencies:

pip install pandas numpy scikit-learn scipy matplotlib seaborn

Run the script:

python imdb_clustering.py

## 📄 License

This project is licensed under the MIT License.
