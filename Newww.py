import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Page Configuration
st.set_page_config(
    page_title="ChatGPT Sentiment Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Text Preprocessing Functions
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Remove negation words from stopwords as they're important for sentiment
        negation_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                         'nowhere', 'none', 'hardly', 'barely', 'scarcely', "don't", 
                         "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't"}
        self.stop_words = self.stop_words - negation_words
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r'\@\w+|\#','', text)
        # Keep important punctuation for sentiment (!, ?)
        text = text.replace('!', ' exclamation ')
        text = text.replace('?', ' question ')
        # Remove other punctuation
        text = text.translate(str.maketrans('', '', string.punctuation.replace('!', '').replace('?', '')))
        # Remove extra digits but keep some context
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 1]
        return ' '.join(tokens)
    
    def preprocess(self, text):
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text

# Sentiment Analysis Model
class SentimentModel:
    def __init__(self):
        # Enhanced TF-IDF with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced from 5000
            ngram_range=(1, 2),
            min_df=3,  # Increased - word must appear in at least 3 documents
            max_df=0.85,  # Reduced - ignore very common words
            sublinear_tf=True,
            use_idf=True
        )
        self.model = None
        self.model_type = None
        self.is_trained = False
    
    def create_sentiment_labels(self, ratings):
        """Create sentiment labels with better distribution"""
        labels = []
        for rating in ratings:
            if rating >= 4:
                labels.append('Positive')
            elif rating <= 2:
                labels.append('Negative')
            else:
                labels.append('Neutral')
        return labels
    
    def add_additional_features(self, X_vec, texts, ratings):
        """Add hand-crafted features to improve performance"""
        from scipy.sparse import hstack
        
        # Create additional features
        additional_features = []
        
        for text, rating in zip(texts, ratings):
            text_str = str(text).lower()
            features = [
                len(text_str),  # Text length
                text_str.count('!'),  # Exclamations
                text_str.count('?'),  # Questions
                text_str.count('exclamation'),  # Processed exclamations
                rating,  # Rating as feature
                # Sentiment indicators
                int('good' in text_str or 'great' in text_str or 'excellent' in text_str or 'amazing' in text_str or 'love' in text_str),
                int('bad' in text_str or 'poor' in text_str or 'terrible' in text_str or 'hate' in text_str or 'worst' in text_str),
                int('not' in text_str or 'never' in text_str or "don't" in text_str or "doesn't" in text_str)
            ]
            additional_features.append(features)
        
        # Convert to sparse matrix and combine
        additional_features = np.array(additional_features)
        return hstack([X_vec, additional_features])
    
    def train(self, X_train, y_train, ratings_train, model_choice='xgboost'):
        """Train with multiple model options"""
        self.model_type = model_choice
        
        # Vectorize text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        # For logistic regression, use simpler features to avoid overfitting
        if model_choice == 'logistic':
            # Use only TF-IDF features for logistic regression
            X_train_enhanced = X_train_vec
        else:
            # Add additional features for tree-based models
            X_train_enhanced = self.add_additional_features(X_train_vec, X_train, ratings_train)
        
        # Choose model with anti-overfitting parameters
        if model_choice == 'xgboost':
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=50,  # Reduced from 100
                learning_rate=0.05,  # Lower learning rate
                max_depth=3,  # Reduced depth to prevent overfitting
                min_samples_split=10,  # Increased
                min_samples_leaf=5,  # Increased
                subsample=0.8,  # Use 80% of samples
                max_features='sqrt',  # Use sqrt of features
                random_state=42
            )
        elif model_choice == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,  # Reduced
                max_depth=15,  # Reduced depth
                min_samples_split=10,  # Increased
                min_samples_leaf=4,  # Increased
                max_features='sqrt',  # Use sqrt of features
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_choice == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                max_iter=200,  # Further reduced
                class_weight='balanced',
                random_state=42,
                C=0.1,  # Much stronger regularization (was 0.5)
                penalty='l2',  # L2 regularization
                solver='saga',  # Better for large datasets
                tol=1e-3  # Earlier stopping
            )
        
        # Train model
        self.model.fit(X_train_enhanced, y_train)
        self.is_trained = True
        self.X_train = X_train  # Store for feature extraction
        self.ratings_train = ratings_train
    
    def predict(self, X_test, ratings_test=None):
        if not self.is_trained:
            return None
        
        X_test_vec = self.vectorizer.transform(X_test)
        
        # For logistic regression, don't add extra features
        if self.model_type == 'logistic':
            X_test_enhanced = X_test_vec
        else:
            # Use dummy ratings if not provided
            if ratings_test is None:
                ratings_test = [3] * len(X_test)
            X_test_enhanced = self.add_additional_features(X_test_vec, X_test, ratings_test)
        
        return self.model.predict(X_test_enhanced)
    
    def predict_proba(self, X_test, ratings_test=None):
        if not self.is_trained:
            return None
        
        X_test_vec = self.vectorizer.transform(X_test)
        
        # For logistic regression, don't add extra features
        if self.model_type == 'logistic':
            X_test_enhanced = X_test_vec
        else:
            # Use dummy ratings if not provided
            if ratings_test is None:
                ratings_test = [3] * len(X_test)
            X_test_enhanced = self.add_additional_features(X_test_vec, X_test, ratings_test)
        
        return self.model.predict_proba(X_test_enhanced)
    
    def get_feature_importance(self, top_n=20):
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            feature_names = list(self.vectorizer.get_feature_names_out())
            feature_names.extend(['text_length', 'exclamations', 'questions', 'proc_exclamations', 
                                 'rating', 'positive_words', 'negative_words', 'negation_words'])
            
            importances = self.model.feature_importances_
            
            # Get top features
            indices = np.argsort(importances)[-top_n:]
            top_features = [(feature_names[i], importances[i]) for i in indices if i < len(feature_names)]
            return sorted(top_features, key=lambda x: x[1], reverse=True)
        return None

# Groq Chatbot Integration
def get_groq_response(user_message, api_key, context_data=None):
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        system_message = f"""You are an AI assistant specializing in sentiment analysis and customer feedback insights. 
        You have access to ChatGPT review data analysis results. Help users understand sentiment patterns, 
        provide insights about customer feedback, and answer questions about the analysis.
        
        Context from analysis:
        {context_data if context_data else 'No specific context provided.'}
        
        Provide clear, concise, and helpful responses."""
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Load and Process Data
@st.cache_data
def load_data(file):
    try:
        # Read Excel file
        df = pd.read_excel(file, engine='openpyxl')
        
        # Display info about loaded data
        st.info(f"📋 Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Check for date column issues
        if 'date' in df.columns:
            # Try to detect if dates are stored as strings or numbers
            sample_dates = df['date'].head()
            st.info(f"Date column sample: {sample_dates.tolist()[:3]}")
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def process_data(df):
    preprocessor = TextPreprocessor()
    
    # Handle text preprocessing
    df['cleaned_review'] = df['review'].apply(preprocessor.preprocess)
    df['cleaned_title'] = df['title'].apply(preprocessor.preprocess)
    df['combined_text'] = df['cleaned_title'] + ' ' + df['cleaned_review']
    
    # Handle date parsing with multiple formats and error handling
    def parse_date_safe(date_val):
        # If already a datetime, return it
        if pd.api.types.is_datetime64_any_dtype(type(date_val)) or isinstance(date_val, datetime):
            return date_val
        
        # If it's a string with ########, return NaT
        if isinstance(date_val, str) and '#' in date_val:
            return pd.NaT
        
        # Try to parse as datetime
        try:
            return pd.to_datetime(date_val, errors='coerce')
        except:
            return pd.NaT
    
    # Apply safe date parsing
    df['date'] = df['date'].apply(parse_date_safe)
    
    # Fill NaT values with a median date from valid dates
    if df['date'].notna().any():
        median_date = df['date'].dropna().median()
        df['date'] = df['date'].fillna(median_date)
    else:
        # If all dates are invalid, use current date
        df['date'] = pd.Timestamp.now()
    
    # Create year_month column
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    
    # Display statistics about date parsing
    valid_dates = df['date'].notna().sum()
    st.info(f"✅ Processed {valid_dates} valid dates out of {len(df)} records")
    
    return df

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = SentimentModel()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main App
def main():
    st.markdown('<h1 class="main-header">💬 ChatGPT Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/6134/6134346.png", width=100)
        st.title("Navigation")
        page = st.radio("Go to", ["📊 Data Upload & Overview", "🔍 Sentiment Analysis", 
                                   "📈 Detailed Insights", "🤖 AI Chatbot Assistant"])
        
        st.markdown("---")
        st.markdown("### Groq API Configuration")
        groq_api_key = st.text_input("Enter Groq API Key", type="password", 
                                     help="Get your API key from https://console.groq.com")
        
        if groq_api_key:
            st.success("✅ API Key configured")
    
    # Page 1: Data Upload & Overview
    if page == "📊 Data Upload & Overview":
        st.header("📤 Upload Dataset")
        
        uploaded_file = st.file_uploader("Upload ChatGPT Reviews Dataset (Excel)", type=['xlsx'])
        
        if uploaded_file:
            df = load_data(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Dataset loaded successfully! Total reviews: {len(df)}")
                
                # Process data
                with st.spinner("Processing data..."):
                    try:
                        df = process_data(df)
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.success("✅ Data processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.info("💡 Tip: Make sure your Excel file has the correct date format. Try opening the file and widening the date column if you see ########")
                        return
                
                # Dataset Overview
                st.subheader("📋 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Reviews", len(df))
                with col2:
                    st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
                with col3:
                    valid_dates = df['date'].notna().sum()
                    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}" if valid_dates > 0 else "N/A"
                    st.metric("Date Range", date_range)
                with col4:
                    st.metric("Languages", df['language'].nunique())
                
                # Display sample data
                st.subheader("📊 Sample Data")
                
                # Add controls for data display
                col1, col2 = st.columns([1, 3])
                with col1:
                    num_rows = st.selectbox("Rows to display", [10, 25, 50, 100, 250], index=0)
                with col2:
                    search_term = st.text_input("🔍 Search in reviews/titles", "")
                
                # Filter data if search term exists
                display_df = df[['date', 'title', 'review', 'rating', 'platform', 'location']].copy()
                
                if search_term:
                    mask = (display_df['title'].str.contains(search_term, case=False, na=False) | 
                           display_df['review'].str.contains(search_term, case=False, na=False))
                    display_df = display_df[mask]
                    st.info(f"Found {len(display_df)} matching reviews")
                
                # Display data with styling
                st.dataframe(
                    display_df.head(num_rows),
                    use_container_width=True,
                    height=400
                )
                
                # Rating Distribution
                st.subheader("⭐ Rating Distribution")
                fig_rating = px.histogram(df, x='rating', nbins=5, 
                                         title='Distribution of Ratings',
                                         color_discrete_sequence=['#667eea'])
                fig_rating.update_layout(xaxis_title='Rating', yaxis_title='Count')
                st.plotly_chart(fig_rating, use_container_width=True)
                
                # Platform & Location Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📱 Platform Distribution")
                    platform_counts = df['platform'].value_counts()
                    fig_platform = px.pie(values=platform_counts.values, 
                                         names=platform_counts.index,
                                         title='Reviews by Platform')
                    st.plotly_chart(fig_platform, use_container_width=True)
                
                with col2:
                    st.subheader("🌍 Top 10 Locations")
                    location_counts = df['location'].value_counts().head(10)
                    fig_location = px.bar(x=location_counts.values, 
                                         y=location_counts.index,
                                         orientation='h',
                                         title='Reviews by Location')
                    fig_location.update_layout(xaxis_title='Count', yaxis_title='Location')
                    st.plotly_chart(fig_location, use_container_width=True)
    
    # Page 2: Sentiment Analysis
    elif page == "🔍 Sentiment Analysis":
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data first from the 'Data Upload & Overview' page")
            return
        
        df = st.session_state.df
        st.header("🎯 Sentiment Classification Model")
        
        # Model selection
        st.subheader("⚙️ Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                "Select Model",
                ["xgboost", "random_forest", "logistic"],
                help="XGBoost usually performs best for text classification"
            )
        
        with col2:
            st.info(f"**Selected:** {model_choice.replace('_', ' ').title()}")
        
        if st.button("🚀 Train Sentiment Model"):
            with st.spinner("Training model... This may take a minute"):
                # Reset model for new training
                st.session_state.model = SentimentModel()
                model = st.session_state.model
                
                # Create sentiment labels
                df['sentiment'] = model.create_sentiment_labels(df['rating'])
                
                # Show class distribution
                sentiment_dist = df['sentiment'].value_counts()
                st.info(f"📊 Class Distribution:\n- Positive: {sentiment_dist.get('Positive', 0)}\n- Neutral: {sentiment_dist.get('Neutral', 0)}\n- Negative: {sentiment_dist.get('Negative', 0)}")
                
                # Filter out rows with empty text
                df_clean = df[df['combined_text'].str.strip() != ''].copy()
                st.info(f"📝 Using {len(df_clean)} reviews with valid text")
                
                # Split data with stratification
                from sklearn.model_selection import train_test_split as tts
                X_train, X_test, y_train, y_test, ratings_train, ratings_test = tts(
                    df_clean['combined_text'], 
                    df_clean['sentiment'],
                    df_clean['rating'],
                    test_size=0.2, 
                    random_state=42, 
                    stratify=df_clean['sentiment']
                )
                
                # Train model with selected algorithm
                model.train(X_train, y_train, ratings_train.values, model_choice=model_choice)
                
                # Predictions on test set
                if model_choice == 'logistic':
                    # Logistic regression doesn't use rating features
                    y_pred = model.predict(X_test, None)
                    y_pred_proba = model.predict_proba(X_test, None)
                    y_train_pred = model.predict(X_train, None)
                else:
                    # Tree-based models use rating features
                    y_pred = model.predict(X_test, ratings_test.values)
                    y_pred_proba = model.predict_proba(X_test, ratings_test.values)
                    y_train_pred = model.predict(X_train, ratings_train.values)
                
                # Calculate training accuracy to check overfitting
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Store results
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.train_accuracy = train_accuracy
                st.session_state.test_accuracy = test_accuracy
                st.session_state.model_choice = model_choice  # Store model choice
                
                
                # Predict for all data
                if model_choice == 'logistic':
                    df['predicted_sentiment'] = model.predict(df['combined_text'], None)
                else:
                    df['predicted_sentiment'] = model.predict(df['combined_text'], df['rating'].values)
                
                st.session_state.df = df
                
                st.success(f"✅ Model trained successfully with {model_choice.replace('_', ' ').title()}!")
                st.info(f"📊 Train Accuracy: {train_accuracy:.2%} | Test Accuracy: {test_accuracy:.2%}")
                st.balloons()
        
        # Display Results
        if st.session_state.model.is_trained:
            st.subheader("📊 Model Performance Metrics")
            
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                st.metric("Recall", f"{recall:.2%}")
            with col4:
                st.metric("F1-Score", f"{f1:.2%}")
            
            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Neutral', 'Negative'])
            
            fig_cm = px.imshow(cm, 
                              labels=dict(x="Predicted", y="Actual", color="Count"),
                              x=['Positive', 'Neutral', 'Negative'],
                              y=['Positive', 'Neutral', 'Negative'],
                              text_auto=True,
                              color_continuous_scale='Blues')
            fig_cm.update_layout(title='Confusion Matrix')
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues'))
            
            # Sentiment Distribution
            st.subheader("📊 Predicted Sentiment Distribution")
            sentiment_counts = df['predicted_sentiment'].value_counts()
            
            fig_sentiment = px.pie(values=sentiment_counts.values,
                                  names=sentiment_counts.index,
                                  title='Overall Sentiment Distribution',
                                  color_discrete_map={'Positive': '#4ade80', 
                                                     'Neutral': '#fbbf24', 
                                                     'Negative': '#f87171'})
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Sentiment by Rating
            st.subheader("⭐ Sentiment vs Rating Analysis")
            sentiment_rating = df.groupby(['rating', 'predicted_sentiment']).size().reset_index(name='count')
            fig_sentiment_rating = px.bar(sentiment_rating, x='rating', y='count', 
                                         color='predicted_sentiment',
                                         title='Sentiment Distribution by Rating',
                                         barmode='group',
                                         color_discrete_map={'Positive': '#4ade80', 
                                                           'Neutral': '#fbbf24', 
                                                           'Negative': '#f87171'})
            st.plotly_chart(fig_sentiment_rating, use_container_width=True)
    

    # Page 3: Detailed Insights
    elif page == "📈 Detailed Insights":
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data first")
            return
        
        df = st.session_state.df
        st.header("Detailed Analysis & Insights")
        st.markdown("---")
        
        # 1. Rating Distribution
        st.subheader("1️⃣ What is the distribution of review ratings?")
        st.markdown("**Insight:** Understand overall sentiment — are users mostly happy or frustrated? 👍👎")
        fig_rating = px.histogram(df, x='rating', nbins=5, 
                                 title='Distribution of Ratings (1 to 5 Stars)',
                                 color_discrete_sequence=['#667eea'],
                                 labels={'rating': 'Rating (Stars)', 'count': 'Number of Reviews'})
        fig_rating.update_layout(xaxis_title='Rating', yaxis_title='Count', bargap=0.1)
        st.plotly_chart(fig_rating, use_container_width=True)
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rating", f"{df['rating'].mean():.2f} ⭐")
        with col2:
            st.metric("Most Common Rating", f"{df['rating'].mode()[0]} ⭐")
        with col3:
            positive_pct = (df['rating'] >= 4).sum() / len(df) * 100
            st.metric("Positive Reviews", f"{positive_pct:.1f}%")
        
        st.markdown("---")
        
        # 2. Helpful Reviews Analysis
        st.subheader("2️⃣ How many reviews were marked as helpful?")
        st.markdown("**Insight:** See how much value users find in reviews (above threshold)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            helpful_threshold = st.slider("Helpful votes threshold", 0, 50, 10)
            helpful_reviews = df[df['helpful_votes'] >= helpful_threshold]
            st.metric("Reviews above threshold", len(helpful_reviews))
            
            if len(helpful_reviews) > 0:
                avg_rating_helpful = helpful_reviews['rating'].mean()
                st.metric("Avg rating of helpful reviews", f"{avg_rating_helpful:.2f}")
            
            # Pie chart for helpful vs not helpful
            helpful_counts = pd.DataFrame({
                'Category': ['Helpful (≥ threshold)', 'Not Helpful (< threshold)'],
                'Count': [len(helpful_reviews), len(df) - len(helpful_reviews)]
            })
        
        with col2:
            fig_helpful_pie = px.pie(helpful_counts, values='Count', names='Category',
                                     title=f'Helpful Reviews Distribution (Threshold: {helpful_threshold})',
                                     color_discrete_sequence=['#4ade80', '#f87171'])
            st.plotly_chart(fig_helpful_pie, use_container_width=True)
        
        st.markdown("---")
        
        # 3. Word Clouds for Positive vs Negative Reviews
        st.subheader("3️⃣ What are the most common keywords in positive vs. negative reviews?")
        st.markdown("**Insight:** Discover what users love or complain about 💚❤️")
        
        if 'combined_text' in df.columns:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ☁️ Positive Reviews (4-5 Stars)")
                positive_reviews = df[df['rating'].isin([4, 5])]['combined_text']
                if len(positive_reviews) > 0:
                    text = ' '.join(positive_reviews)
                    wordcloud = WordCloud(width=800, height=400, 
                                        background_color='white',
                                        colormap='Greens').generate(text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No 4-5 star reviews found")
            
            with col2:
                st.markdown("#### ☁️ Negative Reviews (1-2 Stars)")
                negative_reviews = df[df['rating'].isin([1, 2])]['combined_text']
                if len(negative_reviews) > 0:
                    text = ' '.join(negative_reviews)
                    wordcloud = WordCloud(width=800, height=400, 
                                        background_color='white',
                                        colormap='Reds').generate(text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No 1-2 star reviews found")
        
        st.markdown("---")
        
        # 4. Average Rating Over Time
        st.subheader("4️⃣ How has the average rating changed over time?")
        st.markdown("**Insight:** Track user satisfaction over weeks/months 📅")
        
        rating_time = df.groupby('year_month')['rating'].mean().reset_index()
        fig_rating_time = px.line(rating_time, x='year_month', y='rating',
                                 title='Average Rating Trend Over Time',
                                 markers=True,
                                 labels={'year_month': 'Month', 'rating': 'Average Rating'})
        fig_rating_time.add_hline(y=df['rating'].mean(), line_dash="dash", 
                                 line_color="red", 
                                 annotation_text=f"Overall Average: {df['rating'].mean():.2f}")
        fig_rating_time.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig_rating_time, use_container_width=True)
        
        st.markdown("---")
        
        # 5. Ratings by Location
        st.subheader("5️⃣ How do ratings vary by user location?")
        st.markdown("**Insight:** Identify regional differences in satisfaction or experience 🌍")
        
        location_rating = df.groupby('location').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        location_rating.columns = ['location', 'avg_rating', 'count']
        location_rating = location_rating[location_rating['count'] >= 5].sort_values('avg_rating', ascending=False).head(15)
        
        fig_location = px.bar(location_rating, x='avg_rating', y='location',
                             orientation='h',
                             title='Top 15 Locations by Average Rating (min 5 reviews)',
                             color='avg_rating',
                             text='count',
                             color_continuous_scale='RdYlGn',
                             labels={'avg_rating': 'Average Rating', 'location': 'Location'})
        fig_location.update_traces(texttemplate='%{text} reviews', textposition='outside')
        st.plotly_chart(fig_location, use_container_width=True)
        
        st.markdown("---")
        
        # 6. Platform Comparison
        st.subheader("6️⃣ Which platform (Web vs Mobile) gets better reviews?")
        st.markdown("**Insight:** Helps product teams focus improvements 📱💻")
        
        platform_rating = df.groupby('platform')['rating'].agg(['mean', 'count']).reset_index()
        fig_platform = px.bar(platform_rating, x='platform', y='mean',
                             title='Average Rating by Platform',
                             text='count',
                             color='mean',
                             color_continuous_scale='RdYlGn',
                             labels={'mean': 'Average Rating', 'platform': 'Platform'})
        fig_platform.update_traces(texttemplate='%{text} reviews<br>Avg: %{y:.2f}⭐', textposition='outside')
        fig_platform.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig_platform, use_container_width=True)
        
        # Show winner
        best_platform = platform_rating.loc[platform_rating['mean'].idxmax()]
        st.success(f"🏆 **Winner:** {best_platform['platform']} with {best_platform['mean']:.2f} average rating!")
        
        st.markdown("---")
        
        # 7. Verified vs Non-Verified Users
        st.subheader("7️⃣ Are verified users more satisfied than non-verified ones?")
        st.markdown("**Insight:** Indicates whether loyal/paying users are happier ✅❌")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            verified_rating = df.groupby('verified_purchase')['rating'].agg(['mean', 'count']).reset_index()
            fig_verified = px.bar(verified_rating, x='verified_purchase', y='mean',
                                 title='Average Rating by Verification Status',
                                 color='mean',
                                 text='count',
                                 color_continuous_scale='RdYlGn',
                                 labels={'mean': 'Average Rating', 'verified_purchase': 'Verified Purchase'})
            fig_verified.update_traces(texttemplate='%{text} reviews<br>%{y:.2f}⭐', textposition='outside')
            fig_verified.update_layout(yaxis_range=[0, 5])
            st.plotly_chart(fig_verified, use_container_width=True)
        
        with col2:
            # Pie chart for verified distribution
            verified_counts = df['verified_purchase'].value_counts().reset_index()
            verified_counts.columns = ['Verified Status', 'Count']
            fig_verified_pie = px.pie(verified_counts, values='Count', names='Verified Status',
                                      title='Verified vs Non-Verified Distribution',
                                      color_discrete_sequence=['#4ade80', '#fbbf24'])
            st.plotly_chart(fig_verified_pie, use_container_width=True)
        
        st.markdown("---")
        
        # 8. Review Length by Rating
        st.subheader("8️⃣ What's the average length of reviews per rating category?")
        st.markdown("**Insight:** Shows whether people write longer reviews when they're unhappy or very happy 📝")
        
        if 'review_length' in df.columns:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Bar chart
                length_by_rating = df.groupby('rating')['review_length'].mean().reset_index()
                fig_length_bar = px.bar(length_by_rating, x='rating', y='review_length',
                                    title='Average Review Length by Rating',
                                    color='review_length',
                                    text='review_length',
                                    color_continuous_scale='Blues',
                                    labels={'review_length': 'Average Length (characters)', 'rating': 'Rating'})
                fig_length_bar.update_traces(texttemplate='%{text:.0f} chars', textposition='outside')
                st.plotly_chart(fig_length_bar, use_container_width=True)
            
            with col2:
                # Box plot for distribution
                fig_length_box = px.box(df, x='rating', y='review_length',
                                       title='Review Length Distribution by Rating',
                                       color='rating',
                                       labels={'review_length': 'Review Length (characters)', 'rating': 'Rating'})
                st.plotly_chart(fig_length_box, use_container_width=True)
        else:
            st.warning("⚠️ 'review_length' column not found in data")
        
        st.markdown("---")
        
        # 9. Most Mentioned Words in 1-Star Reviews
        st.subheader("9️⃣ What are the most mentioned words in 1-star reviews?")
        st.markdown("**Insight:** Spot recurring issues or complaints 🔍⚠️")
        
        one_star_reviews = df[df['rating'] == 1]['combined_text']
        
        if len(one_star_reviews) > 0:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Word cloud
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                
                text = ' '.join(one_star_reviews)
                wordcloud = WordCloud(width=800, height=400, 
                                    background_color='white',
                                    colormap='Reds').generate(text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Word Cloud: 1-Star Reviews', fontsize=16, fontweight='bold')
                st.pyplot(fig)
            
            with col2:
                # Top words bar chart
                all_words = ' '.join(one_star_reviews).split()
                word_freq = Counter(all_words).most_common(15)
                
                words_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                fig_words = px.bar(words_df, x='Frequency', y='Word',
                                  orientation='h',
                                  title='Top 15 Words in 1-Star Reviews',
                                  color='Frequency',
                                  text='Frequency',
                                  color_continuous_scale='Reds')
                fig_words.update_traces(textposition='outside')
                st.plotly_chart(fig_words, use_container_width=True)
            
            st.info(f"📊 Total 1-star reviews analyzed: {len(one_star_reviews)}")
        else:
            st.info("No 1-star reviews found in the dataset")
        
        st.markdown("---")
        
        # 10. ChatGPT Version Performance
        st.subheader("🔟 What ChatGPT version received the highest average rating?")
        st.markdown("**Insight:** Evaluate improvement or regression across updates 🧪📊")
        
        version_rating = df.groupby('version')['rating'].agg(['mean', 'count']).reset_index()
        version_rating = version_rating.sort_values('mean', ascending=False)
        
        fig_version = px.bar(version_rating, x='version', y='mean',
                            title='Average Rating by ChatGPT Version',
                            text='count',
                            color='mean',
                            color_continuous_scale='RdYlGn',
                            labels={'mean': 'Average Rating', 'version': 'ChatGPT Version'})
        fig_version.update_traces(texttemplate='%{text} reviews<br>%{y:.2f}⭐', textposition='outside')
        fig_version.update_layout(yaxis_range=[0, 5])
        st.plotly_chart(fig_version, use_container_width=True)
        
        # Show best and worst versions
        col1, col2 = st.columns(2)
        with col1:
            best_version = version_rating.iloc[0]
            st.success(f"🏆 **Best Version:** {best_version['version']} - {best_version['mean']:.2f}⭐ ({best_version['count']} reviews)")
        with col2:
            worst_version = version_rating.iloc[-1]
            st.error(f"⚠️ **Lowest Rated:** {worst_version['version']} - {worst_version['mean']:.2f}⭐ ({worst_version['count']} reviews)")
        
    
    # Page 4: AI Chatbot
    elif page == "🤖 AI Chatbot Assistant":
        st.header("🤖 AI-Powered Insights Assistant")
        st.markdown("Ask questions about the sentiment analysis results!")
        
        if not groq_api_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar to use the chatbot")
            return
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload and analyze data first to get contextual insights")
            return
        
        df = st.session_state.df
        
        # Prepare context
        context = f"""
        Dataset Summary:
        - Total Reviews: {len(df)}
        - Average Rating: {df['rating'].mean():.2f}
        - Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
        - Platforms: {', '.join(df['platform'].unique())}
        - Top Locations: {', '.join(df['location'].value_counts().head(5).index.tolist())}
        """
        
        if 'predicted_sentiment' in df.columns:
            sentiment_dist = df['predicted_sentiment'].value_counts().to_dict()
            context += f"\n- Sentiment Distribution: {sentiment_dist}"
        
        # Chat interface
        st.markdown("### 💭 Chat History")
        
        # Display chat history
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"**🧑 You:** {message}")
            else:
                st.markdown(f"**🤖 Assistant:** {message}")
        
        # User input
        user_input = st.text_input("Ask a question about the analysis:", key="user_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_button = st.button("Send 📤")
        with col2:
            if st.button("Clear History 🗑️"):
                st.session_state.chat_history = []
                st.rerun()
        
        if send_button and user_input:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_input))
            
            # Get AI response
            with st.spinner("Thinking..."):
                response = get_groq_response(user_input, groq_api_key, context)
            
            # Add assistant response to history
            st.session_state.chat_history.append(("assistant", response))
            
            st.rerun()
        
        # Suggested questions
        st.markdown("### 💡 Suggested Questions")
        suggestions = [
            "What are the key insights from the sentiment analysis?",
            "Which platform has better user satisfaction?",
            "What are common complaints in negative reviews?",
            "How has sentiment changed over time?",
            "What improvements should be prioritized?"
        ]
        
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion}"):
                st.session_state.chat_history.append(("user", suggestion))
                with st.spinner("Thinking..."):
                    response = get_groq_response(suggestion, groq_api_key, context)
                st.session_state.chat_history.append(("assistant", response))
                st.rerun()

if __name__ == "__main__":
    main()