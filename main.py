import streamlit as st
import pandas as pd
from bertopic import BERTopic

# Embeddings
from sentence_transformers import SentenceTransformer
# from transformers.pipelines import pipeline
# from flair.embeddings import TransformerDocumentEmbeddings
# from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
# import spacy
# import tensorflow_hub
# import gensim.downloader as api

# Dimensionality reduction
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# from cuml.manifold import UMAP
from bertopic.dimensionality import BaseDimensionalityReduction

# Clustering
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
# from cuml.cluster import HDBSCAN

# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer
# from bertopic.vectorizers import OnlineCountVectorizer

# c-TF-IDF
from bertopic.vectorizers import ClassTfidfTransformer

# Fine-tuning
import openai
import tiktoken
from bertopic.representation import TextGeneration
from bertopic.representation import OpenAI

st.title("BERTopic Explorer")
st.write("Explore topics in datasets using the BERTopic model.")


def load_data():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None


def select_data_column(df):
    if df is not None:
        selected_column = st.selectbox("Select a column for analysis", df.columns)
        return df[selected_column].astype(str).tolist()
    return []


def model_settings():
    st.sidebar.header("Model Settings")

    # Embedding Model Selection
    st.sidebar.subheader("Embedding Model Selection")
    embedding_model_name = st.sidebar.selectbox(
        "Choose Embedding Model",
        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "other-model-name"],
        key="embedding_model_select"
    )
    embedding_model = SentenceTransformer(embedding_model_name)

    # Dimensionality Reduction Settings
    st.sidebar.subheader("Dimensionality Reduction Settings")
    dimensionality_model_name = st.sidebar.selectbox(
        "Choose Dimensionality Reduction Model",
        ["UMAP", "PCA", "Truncated SVD", "None"],
        key="dimensionality_model_select"
    )

    if dimensionality_model_name == "UMAP":
        n_neighbors = st.sidebar.slider("UMAP n_neighbors", 5, 50, 15)
        n_components = st.sidebar.slider("UMAP n_components", 2, 10, 5)
        min_dist = st.sidebar.slider("UMAP min_dist", 0.0, 1.0, 0.0)
        umap_metric = st.sidebar.selectbox("UMAP Metric", ["cosine", "euclidean", "manhattan"])
        dimensionality_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=umap_metric
        )

    elif dimensionality_model_name == "PCA":
        n_components_pca = st.sidebar.slider("PCA n_components", 2, 100, 50)
        dimensionality_model = PCA(n_components=n_components_pca)

    elif dimensionality_model_name == "Truncated SVD":
        n_components_svd = st.sidebar.slider("SVD n_components", 2, 100, 50)
        dimensionality_model = TruncatedSVD(n_components=n_components_svd)

    elif dimensionality_model_name == "None":
        empty_dimensionality_model = BaseDimensionalityReduction()
        dimensionality_model = empty_dimensionality_model

    # Clustering Settings
    st.sidebar.subheader("Clustering Settings")
    clustering_model_name = st.sidebar.selectbox(
        "Choose Clustering Model",
        ["HDBSCAN", "k-Means", "Agglomerative Clustering", "cuML HDBSCAN"],
        key="clustering_model_select"
    )

    if clustering_model_name == "HDBSCAN":
        min_cluster_size = st.sidebar.slider("HDBSCAN min_cluster_size", 5, 100, 15)
        hdbscan_metric = st.sidebar.selectbox("HDBSCAN Metric", ["euclidean", "manhattan"])
        clustering_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=hdbscan_metric,
            cluster_selection_method='eom',
            prediction_data=True
        )

    elif clustering_model_name == "k-Means":
        n_clusters = st.sidebar.slider("k-Means n_clusters", 2, 100, 8)
        clustering_model = KMeans(n_clusters=n_clusters)

    elif clustering_model_name == "Agglomerative Clustering":
        n_clusters_agg = st.sidebar.slider("Agglomerative Clustering n_clusters", 2, 100, 8)
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters_agg)

    elif clustering_model_name == "cuML HDBSCAN":
        min_samples = st.sidebar.slider("cuML HDBSCAN min sample size", 5, 100, 15)
        clustering_model = HDBSCAN(
            min_samples=min_samples,
            gen_min_span_tree=True,
            prediction_data=True
        )
    # Vectorization Settings
    st.sidebar.subheader("Vectorizer Settings")
    vectorizer_type = st.sidebar.selectbox(
        "Choose Vectorizer",
        ["CountVectorizer", "OnlineCountVectorizer"],
        key="Vectorizer_model_select"
    )
    if vectorizer_type == "CountVectorizer":
        ngram_min, ngram_max = st.sidebar.slider("Ngram Range", 1, 5, (1, 2))
        ngram_range = (int(ngram_min), int(ngram_max))
        remove_stop_words = st.sidebar.checkbox("Remove Stopwords", value=False)
        stop_words = "english" if remove_stop_words else None  # Bit convoluted
        vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)

    elif vectorizer_type == "OnlineCountVectorizer":
        pass  # Work in progress

    # c-TF-IDF parameters
    st.sidebar.subheader("c-TF-IDF Settings")
    bm25_weighting = st.sidebar.checkbox("BM25 Weighting", value=True)
    reduce_frequent_words = st.sidebar.checkbox("Reduce Frequent Words", value=False)
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=bm25_weighting, reduce_frequent_words=reduce_frequent_words)

    # Add Fine-Tuning representations section
    # add Open AI integration
    st.sidebar.subheader("Fine-Tuning Settings")
    fine_tuning_model_type = st.sidebar.selectbox(
        "Choose Fine-Tuning Model",
        ["None", "OpenAI"],
        key="fine_tuning_model_select"
    )
    if fine_tuning_model_type == "OpenAI":
        text_generation_model = st.sidebar.selectbox(
            "Choose Text Generation Model",
            ["gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"],
            key="text_generation_model_select"
        )
        # Add some option widgets to the sidebar:
        # Add secrets
        api_key = st.sidebar.text_input("API Key", type="password")

        # Add chat option
        chat = st.sidebar.checkbox("Enable Chat (Set this to True if a GPT-3.5 model is used)", value=True)

        # Add custom prompt option
        custom_prompt = st.sidebar.checkbox("Custom Prompt", value=False)
        prompt = None
        if custom_prompt:
            prompt = st.sidebar.text_area(
                "Modify Prompt", "I have topic that contains the following documents:\n[DOCUMENTS]\nThe topic is described by the following keywords:\n[KEYWORDS]\n\nBased on the information above, extract a short topic label in the following format:\ntopic: <topic label>")
            st.sidebar.code(prompt)

        # Configure OpenAI client
        tokenizer = tiktoken.get_encoding("cl100k_base")
        client = openai.OpenAI(api_key=api_key)
        representation_model = OpenAI(
            client,
            model=text_generation_model,
            delay_in_seconds=2,
            chat=chat,
            nr_docs=4,
            doc_length=100,
            tokenizer=tokenizer
        )

    elif fine_tuning_model_type == "None":
        representation_model = None

    return {
        "embedding_model": embedding_model,
        "dimensionality_model": dimensionality_model,
        "clustering_model": clustering_model,
        "vectorizer_model": vectorizer_model,
        "ctfidf_model": ctfidf_model,
        "representation_model": representation_model
    }


def run_bertopic(docs, settings):
    if not docs:
        st.error("No documents for analysis. Please upload a file and select a column.")
        return

    # Get settings from model settings function
    embedding_model = settings["embedding_model"]
    dimensionality_model = settings["dimensionality_model"]
    clustering_model = settings["clustering_model"]
    vectorizer_model = settings["vectorizer_model"]
    ctfidf_model = settings["ctfidf_model"]
    representation_model = settings["representation_model"]

    # Run BERTopic
    topic_model = BERTopic(embedding_model=embedding_model,
                           umap_model=dimensionality_model,
                           hdbscan_model=clustering_model,
                           vectorizer_model=vectorizer_model,
                           ctfidf_model=ctfidf_model,
                           representation_model=representation_model,
                           verbose=True)

    topics, probs = topic_model.fit_transform(docs)
    st.session_state['model'] = topic_model
    st.session_state['topics'] = topics
    st.session_state['probs'] = probs

    return topic_model


def generate_graphs(topic_model):
    if topic_model:
        try:
            # Set up graphs
            topic_map = topic_model.visualize_topics()
            topic_distribution = topic_model.visualize_distribution(st.session_state['probs'])
            barchart = topic_model.visualize_barchart(top_n_topics=10)

            # Set graph title font color to white
            style_update = {'title_font_color': 'white'}
            topic_map.update_layout(**style_update)
            topic_distribution.update_layout(**style_update)
            barchart.update_layout(**style_update)

            # Plot graphs
            st.plotly_chart(topic_map, use_container_width=True)
            st.plotly_chart(topic_distribution, use_container_width=True)
            st.plotly_chart(barchart, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating graphs: {e}")


df = load_data()
if df is not None:
    # Override a function to avoid slow display of pd.Style
    st.elements.lib.pandas_styler_utils._use_display_values = lambda df, style: df.astype(str)
    # Display df
    st.dataframe(df)

docs = select_data_column(df)

settings = model_settings()

if st.button("Run BERTopic"):
    with st.spinner('Running BERTopic...'):
        model = run_bertopic(docs, settings)
    st.success('Topic modeling completed!')

if 'model' in st.session_state:
    topic_info = st.session_state['model'].get_topic_info()
    st.write(topic_info)

if st.button("Generate Visual Statistics") and 'model' in st.session_state:
    generate_graphs(st.session_state['model'])
