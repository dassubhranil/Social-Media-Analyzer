import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
import praw
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import re
import spacy
from datetime import datetime
from transformers import pipeline
from sentence_transformers import SentenceTransformer


# -------------------- Streamlit Page Configuration --------------------
st.set_page_config(
    page_title="Social Media Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS for Styling --------------------
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        width: 100%;
    }
    .stTitle {
        color: #ff4b4b;
    }
    .stHeader {
        color: #262730;
    }
    .footer {
        text-align: center;
        padding: 10px;
        color: #888;
    }
    .footer a {
        color: #ff4b4b;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer svg {
        vertical-align: middle;
    }
    /* Reduce font size for metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }
    /* Make tabs horizontally scrollable on smaller screens */
    div[role="tablist"] {
        overflow-x: auto;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Initialize Session State --------------------
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'df1' not in st.session_state:
    st.session_state.df1 = pd.DataFrame()
if 'df2' not in st.session_state:
    st.session_state.df2 = pd.DataFrame()
if 'keyword1' not in st.session_state:
    st.session_state.keyword1 = "NVIDIA"
if 'keyword2' not in st.session_state:
    st.session_state.keyword2 = "AMD"
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Single Keyword"


# -------------------- Main Title --------------------
st.title("✨Social Media Sentiment & Trend Analyzer")
st.markdown("Unlock deeper insights with Emotion Analysis, Sarcasm Detection, and Side-by-Side Keyword Comparison.")

# -------------------- Sidebar for User Inputs --------------------
with st.sidebar:
    st.header("⚙️ Search Parameters")
    
    # Analysis Mode Selection
    analysis_mode = st.radio(
        "Select Analysis Mode",
        ("Single Keyword", "Keyword Comparison"),
        horizontal=True,
        key='analysis_mode_selector'
    )
    
    if analysis_mode == "Single Keyword":
        keyword1 = st.text_input("Enter Keyword", value=st.session_state.keyword1)
        keyword2 = None
    else: # Keyword Comparison
        col1, col2 = st.columns(2)
        with col1:
            keyword1 = st.text_input("Enter Keyword 1", value=st.session_state.keyword1)
        with col2:
            keyword2 = st.text_input("Enter Keyword 2", value=st.session_state.keyword2)

    num_posts = st.slider("Number of items to analyze (per keyword)", 50, 1000, 250)

    st.markdown("---")
    st.header("💡 Advanced Features", help="Uses advanced models and may be slower.")
    enable_ner = st.toggle("Entity Sentiment Analysis (NER)", value=True)
    enable_dtm = st.toggle("Dynamic Topic Modeling (DTM)", value=True)
    enable_emotion_sarcasm = st.toggle("Emotion & Sarcasm Analysis", value=True")
    
    st.markdown("---")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        start_analysis_button = st.button("Start Analysis")
    with col_b2:
        reset_button = st.button("Reset")

    st.markdown("---")
    st.info("This app uses Reddit credentials stored securely via Streamlit Secrets.", icon="🔐")


# -------------------- Model Loading (Cached) --------------------

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model from the installed package."""
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_transformer_pipelines():
    """Loads Hugging Face transformer models for emotion and sarcasm."""
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    sarcasm_detector = pipeline("text-classification", model="helinivan/english-sarcasm-detector")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return emotion_classifier, sarcasm_detector, embedding_model

nlp = load_spacy_model()
if enable_emotion_sarcasm:
    emotion_classifier, sarcasm_detector, embedding_model = load_transformer_pipelines()
else:
    # Still need the embedding model for BERTopic
    @st.cache_resource
    def load_embedding_model():
        return SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model = load_embedding_model()


# -------------------- Helper Functions --------------------

@st.cache_data(ttl=600)
def fetch_reddit_data(_reddit_instance, keyword, limit):
    """Fetches posts and their top comments from Reddit until the limit is reached."""
    data = []
    last_post_id = None
    
    try:
        while len(data) < limit:
            # Fetch a batch of posts
            params = {'after': last_post_id} if last_post_id else {}
            posts = list(_reddit_instance.subreddit('all').search(keyword, limit=100, params=params))
            
            if not posts:
                break # No more posts to fetch
            
            for post in posts:
                # Add post to data
                if len(data) < limit:
                    data.append({'text': post.title + " " + post.selftext, 'timestamp': datetime.utcfromtimestamp(post.created_utc)})
                
                # Fetch comments for the post
                post.comments.replace_more(limit=0) # Remove "load more comments"
                for comment in post.comments.list():
                    if len(data) < limit:
                        data.append({'text': comment.body, 'timestamp': datetime.utcfromtimestamp(comment.created_utc)})
                    else:
                        break
                
                if len(data) >= limit:
                    break
            
            last_post_id = posts[-1].fullname

        if not data:
            st.warning(f"No content found for '{keyword}'. Try another keyword.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    except Exception as e:
        st.error(f"Failed to fetch data for '{keyword}': {e}")
        return pd.DataFrame()


def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()

@st.cache_data
def run_full_analysis(_df, keyword):
    """Performs all analyses on the dataframe."""
    _df['cleaned_text'] = _df['text'].apply(clean_text)
    
    # 1. VADER Sentiment
    analyzer = SentimentIntensityAnalyzer()
    _df['sentiment_score'] = _df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    # 2. Sarcasm Detection (if enabled)
    if enable_emotion_sarcasm:
        # Truncate text for model
        _df['sarcasm_label'] = _df['cleaned_text'].apply(lambda x: sarcasm_detector(x[:512])[0]['label'])
        # Adjust sentiment for sarcastic posts
        _df['adjusted_sentiment_score'] = _df.apply(
            lambda row: -row['sentiment_score'] if row['sarcasm_label'] == 'sarcastic' and row['sentiment_score'] != 0 else row['sentiment_score'],
            axis=1
        )
        sentiment_col = 'adjusted_sentiment_score'
    else:
        sentiment_col = 'sentiment_score'

    _df['sentiment_label'] = _df[sentiment_col].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

    # 3. Emotion Analysis (if enabled)
    if enable_emotion_sarcasm:
        emotions = _df['cleaned_text'].apply(lambda x: pd.Series({e['label']: e['score'] for e in emotion_classifier(x[:512])[0]}))
        _df = pd.concat([_df, emotions], axis=1)

    return _df

# -------------------- Main Application Logic --------------------

def display_dashboard(keyword, df):
    """Generates and displays the entire analysis dashboard for a given keyword."""
    st.header(f"📊 Analysis for '{keyword}'")

    if df.empty:
        st.warning(f"No data available to analyze for '{keyword}'.")
        return

    # Run analyses
    with st.spinner(f"Running advanced analysis on {len(df)} items for '{keyword}'..."):
        df_analyzed = run_full_analysis(df.copy(), keyword)
        topic_model, topic_info = perform_topic_modeling(df_analyzed, keyword)

    # Wrap dashboard content in a scrollable container
    with st.container(height=1000):
        # --- Display Results ---
        st.subheader("Key Metrics")
        sentiment_col = 'adjusted_sentiment_score' if enable_emotion_sarcasm else 'sentiment_score'
        avg_sentiment = df_analyzed[sentiment_col].mean()
        sentiment_counts = df_analyzed['sentiment_label'].value_counts()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Items", len(df_analyzed))
        c2.metric("Avg. Sentiment", f"{avg_sentiment:.2f}")
        c3.metric("Dominant Sentiment", sentiment_counts.idxmax())
        if topic_model:
            c4.metric("Topics Discovered", len(topic_info[topic_info.Topic != -1]))
        else:
            c4.metric("Topics Discovered", 0)
        
        # Setup tabs
        tabs_list = ["Sentiment Analysis"]
        if enable_emotion_sarcasm: tabs_list.append("Emotion & Sarcasm")
        tabs_list.extend(["Word Clouds", "Topic Modeling"])
        if enable_ner: tabs_list.append("Entity Analysis")
        if enable_dtm and topic_model: tabs_list.append("Topic Evolution")
        
        tabs = st.tabs(tabs_list)
        tab_map = {name: tab for name, tab in zip(tabs_list, tabs)}

        with tab_map["Sentiment Analysis"]:
            fig_pie = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, title='Sentiment Proportions', color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'grey'})
            st.plotly_chart(fig_pie, use_container_width=True, key=f"sentiment_pie_{keyword}")

        if "Emotion & Sarcasm" in tab_map:
            with tab_map["Emotion & Sarcasm"]:
                st.subheader("Emotion & Sarcasm Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    emotion_cols = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
                    emotion_means = df_analyzed[emotion_cols].mean().sort_values(ascending=False)
                    fig_emotion = px.bar(emotion_means, x=emotion_means.index, y=emotion_means.values, title="Average Emotion Distribution", labels={'x': 'Emotion', 'y': 'Average Score'})
                    st.plotly_chart(fig_emotion, use_container_width=True, key=f"emotion_bar_{keyword}")
                with col2:
                    sarcasm_counts = df_analyzed['sarcasm_label'].value_counts()
                    fig_sarcasm = px.pie(sarcasm_counts, names=sarcasm_counts.index, values=sarcasm_counts.values, title='Sarcasm Detection')
                    st.plotly_chart(fig_sarcasm, use_container_width=True, key=f"sarcasm_pie_{keyword}")

        with tab_map["Word Clouds"]:
            selected_sentiment = st.selectbox(f"Select Sentiment for '{keyword}'", ['Positive', 'Negative', 'Neutral'], key=f"wc_select_{keyword}")
            
            # Add a theme-aware title using st.subheader
            st.subheader(f"Word Cloud for '{selected_sentiment}' Sentiment")

            text_data = " ".join(df_analyzed[df_analyzed['sentiment_label'] == selected_sentiment]['cleaned_text'].tolist())
            if text_data:
                custom_stopwords = set(STOPWORDS); custom_stopwords.add(keyword.lower())
                wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords, colormap='viridis').generate(text_data)
                
                # Create the figure with a specific size
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                
                # Display the plot, making it responsive
                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.warning(f"No text data available for '{selected_sentiment}' sentiment in '{keyword}' posts.")

        with tab_map["Topic Modeling"]:
            if topic_model and not topic_info.empty:
                display_topics = topic_info[topic_info.Topic != -1].head(10)
                st.dataframe(display_topics[['Topic', 'Name', 'Count']].rename(columns={'Name': 'Keywords', 'Count': 'Freq.'}))
            else:
                st.info("Topic modeling could not be performed.")

        if "Entity Analysis" in tab_map:
            with tab_map["Entity Analysis"]:
                with st.spinner("Analyzing entities..."):
                    entity_df = analyze_entities_sentiment(df_analyzed, keyword)
                if not entity_df.empty:
                    fig_ent = px.bar(entity_df.head(15), x='Entity', y='Average_Sentiment', color='Average_Sentiment', color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[-1, 1], title="Sentiment per Entity")
                    st.plotly_chart(fig_ent, use_container_width=True, key=f"entity_bar_{keyword}")
                else:
                    st.warning("No significant entities found.")
        
        if "Topic Evolution" in tab_map:
            with tab_map["Topic Evolution"]:
                with st.spinner("Analyzing topic trends..."):
                    topics_over_time = perform_dynamic_topic_modeling(topic_model, df_analyzed, keyword)
                if topics_over_time is not None:
                    fig_dtm = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
                    # Update tooltip for better readability in both light and dark modes
                    fig_dtm.update_layout(
                        hoverlabel=dict(
                            bgcolor="lightgrey",
                            font_color="black"
                        )
                    )
                    st.plotly_chart(fig_dtm, use_container_width=True, key=f"dtm_plot_{keyword}")
                else:
                    st.warning("Could not generate dynamic topic model.")

# --- Functions needed for the dashboard (placed here for clarity) ---
@st.cache_data
def perform_topic_modeling(_df, keyword):
    docs = _df['cleaned_text'].tolist()
    if len(docs) < 15: return None, pd.DataFrame()
    try:
        topic_model = BERTopic(
            embedding_model=embedding_model, # Use the pre-loaded model
            vectorizer_model=CountVectorizer(stop_words="english"), 
            calculate_probabilities=True, 
            verbose=False
        )
        topics, _ = topic_model.fit_transform(docs)
        return topic_model, topic_model.get_topic_info()
    except Exception: return None, pd.DataFrame()

@st.cache_data
def perform_dynamic_topic_modeling(_topic_model, _df, keyword):
    if _topic_model is None or _df.empty: return None
    try:
        # Group timestamps into 20 bins to improve performance
        return _topic_model.topics_over_time(_df['cleaned_text'].tolist(), _df['timestamp'].tolist(), nr_bins=20)
    except Exception: return None

@st.cache_data
def analyze_entities_sentiment(_df, keyword):
    entity_sentiments = {}
    sentiment_col = 'adjusted_sentiment_score' if 'adjusted_sentiment_score' in _df.columns else 'sentiment_score'
    for _, row in _df.iterrows():
        doc = nlp(row['text'])
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "PRODUCT", "GPE"]:
                entity_text = ent.text.strip().lower()
                if entity_text not in entity_sentiments: entity_sentiments[entity_text] = []
                entity_sentiments[entity_text].append(row[sentiment_col])
    avg_entity_sentiments = {e: sum(s)/len(s) for e, s in entity_sentiments.items() if len(s) > 2}
    if not avg_entity_sentiments: return pd.DataFrame()
    sorted_entities = sorted(avg_entity_sentiments.items(), key=lambda item: item[1], reverse=True)
    return pd.DataFrame(sorted_entities, columns=['Entity', 'Average_Sentiment'])


# -------------------- App Execution --------------------
if reset_button:
    # Explicitly reset all session state variables to their defaults
    st.session_state.analysis_run = False
    st.session_state.df1 = pd.DataFrame()
    st.session_state.df2 = pd.DataFrame()
    st.session_state.keyword1 = "NVIDIA"
    st.session_state.keyword2 = "AMD"
    st.session_state.analysis_mode = "Single Keyword"
    # Clear caches
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

if start_analysis_button:
    # Clear previous results and cache to force a fresh run
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.analysis_run = False
    st.session_state.df1 = pd.DataFrame()
    st.session_state.df2 = pd.DataFrame()
    
    try:
        reddit = praw.Reddit(
            client_id=st.secrets["client_id"],
            client_secret=st.secrets["client_secret"],
            user_agent=st.secrets["user_agent"]
        )
        reddit.user.me()
    except Exception:
        st.error("Failed to authenticate with Reddit. Please check your credentials in the Streamlit Secrets.")
        st.stop()

    # Store all parameters from this run in session state
    st.session_state.analysis_mode = analysis_mode
    st.session_state.keyword1 = keyword1
    st.session_state.keyword2 = keyword2

    # Fetch data for keyword(s) and store in session state
    with st.spinner(f"Fetching {num_posts} items for '{keyword1}'..."):
        st.session_state.df1 = fetch_reddit_data(reddit, keyword1, num_posts)
    if analysis_mode == "Keyword Comparison" and keyword2:
        with st.spinner(f"Fetching {num_posts} items for '{keyword2}'..."):
            st.session_state.df2 = fetch_reddit_data(reddit, keyword2, num_posts)
    else:
        st.session_state.df2 = pd.DataFrame()

    st.session_state.analysis_run = True

if st.session_state.analysis_run:
    # Display dashboards using data from session state
    if st.session_state.analysis_mode == "Keyword Comparison" and not st.session_state.df2.empty:
        dash_col1, dash_col2 = st.columns(2)
        with dash_col1:
            display_dashboard(st.session_state.keyword1, st.session_state.df1)
        with dash_col2:
            display_dashboard(st.session_state.keyword2, st.session_state.df2)
    else:
        display_dashboard(st.session_state.keyword1, st.session_state.df1)

# -------------------- Footer --------------------
st.markdown("---")

# Replace with your actual profile URLs
LINKEDIN_URL = "https://www.linkedin.com/in/subhranil-das/"
GITHUB_URL = "https://github.com/dassubhranil"

# SVG Icons
linkedin_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-linkedin"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>"""
github_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-github"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>"""

footer_html = f"""
<div class="footer">
    <p>Built by Subhranil Das</p>
    <p>Connect with me:</p>
    <a href="{LINKEDIN_URL}" target="_blank">{linkedin_svg} LinkedIn</a>
    <a href="{GITHUB_URL}" target="_blank">{github_svg} GitHub</a>
    <p>                                                        </p>
    <p>© 2025 All rights reserved.</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
