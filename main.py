import re
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

import numpy as np
from transformers import CamembertModel, CamembertTokenizer ,CamembertConfig


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import torch

# Loading CamemBERT model config and tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
config = CamembertConfig.from_pretrained("camembert/camembert-base", output_hidden_states=True)
model = CamembertModel.from_pretrained("camembert/camembert-base",config=config)
model.eval()

stop_words_en = stopwords.words()
stop_words_fr = stopwords.words('french')
stop_words = stop_words_en + stop_words_fr 

dataset_path = "./assets/prod.csv"
df = pd.read_csv(dataset_path)

def process_text(value):
    try:
         # Enleve les nombres
        value = re.sub(r'\d','',value)
        # enleve la ponctuation , espaces et converti en minuscule
        value = value.encode('utf-8', 'ignore').decode('utf-8')
        value = re.sub(r'\s+', ' ', value.strip()
                                .lower()
                                .translate(str.maketrans(string.punctuation,
                                                         ' ' * len(string.punctuation))))
        # Enleve les stop word et les rassemble
        words = value.split(' ')
        value = ' '.join(w for w in words if (not w in stop_words) and (len(w) > 2))  
    except Exception as e:
        print(f"Error: {e}")
    
    return value   

# Tokenization et génération des embeddings
def get_embeddings(texts):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings) 

df['Text'] = df['Title'].apply(lambda x: process_text(x)) # Nous analysons les titres des article car les article sont trop lourd a analyser
texts = df['Text'].tolist() # nous convertissons les titres en liste pour facilité la reduction des dimension et la lecture

embeddings = get_embeddings(texts) #Tokenize !

# Réduction de la dimensionnalité pour la visualisation
pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(embeddings)

tsne = TSNE(n_components=2)
vis_embeddings = tsne.fit_transform(reduced_embeddings)

# Clustering avec K-Means
n_clusters = 12
kmeans = KMeans(n_clusters=n_clusters)
clusters = kmeans.fit_predict(reduced_embeddings)

# Analyser les clusters pour identifier les sujets principaux
cluster_subjects = {}
for i in range(n_clusters):
    cluster_texts = [texts[j] for j in range(len(texts)) if clusters[j] == i]
    cluster_subjects[i] = cluster_texts

# Visualisation des clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(vis_embeddings[:, 0], vis_embeddings[:, 1], c=clusters, cmap='viridis')

# for i, text in enumerate(texts): # Affiche le contenu du titres a coté des point ( illisible mais sert d'indication )
#     plt.annotate(text, (vis_embeddings[i, 0], vis_embeddings[i, 1]))

plt.colorbar(scatter, label='Cluster')
plt.title("Visualisation des clusters de textes")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# Affichage des sujets principaux par cluster dans le terminale
for cluster_id, texts in cluster_subjects.items():
    print(f"Cluster {cluster_id}:")
    for text in texts[:8]:  # Afficher les 8 premiers textes de chaque cluster pour avoir une idée globale
        print(f" - {text}")
    print("\n")

# Affichage des cluster axe X et Y
plt.show()

