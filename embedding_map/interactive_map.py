import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import plotly.express as px

# Let's assume `embeddings` is a NumPy array of your text embeddings from the GPT API

# Step 2: Apply t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=42)
reduced_embeddings = tsne.fit_transform(embeddings)

# Step 3: Use DBSCAN to estimate density and find clusters (high-density areas)
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(reduced_embeddings)

# Step 4: Extract keywords from the abstracts using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(abstracts)  # `abstracts` is a list of text abstracts
feature_array = np.array(vectorizer.get_feature_names())
tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]

n = 5  # Number of keywords you want
top_n = feature_array[tfidf_sorting][:n]

# Step 5: Create interactive visualization with Plotly
fig = px.scatter(
    x=reduced_embeddings[:,0], 
    y=reduced_embeddings[:,1], 
    color=clusters,
    hover_name=top_n  # Assumes `top_n` is a list of keywords for each point
)

fig.update_layout(clickmode='event+select')
fig.update_traces(marker_size=10)

# Add custom JavaScript for interactivity (if using in a web app with Dash)
# This is just a placeholder, actual JavaScript code would depend on the use case
fig.update_layout(
    hovermode='closest',
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Hover on points",
                          method="restyle",
                          args=[{"visible": [True, True, False]}])],
        )
    ]
)

fig.show()

