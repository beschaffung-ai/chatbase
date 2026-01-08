import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import re
from typing import Dict, List, Tuple

class ChatAnalyzer:
    def __init__(self, conv_df: pd.DataFrame, msg_df: pd.DataFrame):
        self.conv_df = conv_df
        self.msg_df = msg_df
        self._prepare_data()

    def _prepare_data(self):
        # Filter for user messages for content analysis
        self.user_msgs = self.msg_df[self.msg_df['role'] == 'user'].copy()
        
        # Ensure dates are datetime
        self.conv_df['date'] = pd.to_datetime(self.conv_df['date'])
        
        # Clean up potentially existing columns from previous runs/filtering
        for col in ['content', 'cluster', 'sentiment']:
            if col in self.conv_df.columns:
                self.conv_df = self.conv_df.drop(col, axis=1)

        # Aggregate user text per conversation for clustering
        self.conv_text = self.user_msgs.groupby('conversation_id')['content'].apply(lambda x: ' '.join(x)).reset_index()
        self.conv_df = self.conv_df.merge(self.conv_text, on='conversation_id', how='left')
        self.conv_df['content'] = self.conv_df['content'].fillna('')

        # Download nltk resources if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.german_stopwords = stopwords.words('german')
        # Add custom stopwords relevant to chat context
        self.custom_stopwords = self.german_stopwords + ['hallo', 'hi', 'danke', 'bitte', 'ja', 'nein', 'ok', 'okay', 'guten', 'tag']

    def get_basic_stats(self) -> Dict:
        """Returns descriptive statistics"""
        total_conv = len(self.conv_df)
        total_msgs = len(self.msg_df)
        avg_duration = self.conv_df['duration_seconds'].mean()
        avg_msgs_per_conv = self.msg_df.groupby('conversation_id').size().mean()
        
        return {
            'total_conversations': total_conv,
            'total_messages': total_msgs,
            'avg_duration_seconds': round(avg_duration, 2) if not np.isnan(avg_duration) else 0,
            'avg_messages_per_conversation': round(avg_msgs_per_conv, 2)
        }

    def get_time_distribution(self, freq='D') -> pd.DataFrame:
        """Returns conversation counts aggregated by time frequency (D=Daily, W=Weekly, M=Monthly)"""
        return self.conv_df.set_index('date').resample(freq).size().reset_index(name='count')

    def get_heatmap_data(self) -> pd.DataFrame:
        """Returns a pivot table for Weekday vs Hour heatmap"""
        df = self.conv_df.copy()
        df['weekday'] = df['date'].dt.day_name()
        # Order weekdays
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['weekday'] = pd.Categorical(df['weekday'], categories=cats, ordered=True)
        df['hour'] = df['date'].dt.hour
        
        heatmap_data = df.groupby(['weekday', 'hour']).size().reset_index(name='count')
        return heatmap_data.pivot(index='weekday', columns='hour', values='count').fillna(0)

    def get_exit_analysis(self) -> pd.DataFrame:
        """
        Analyzes the last message of each conversation.
        Returns a DataFrame with last message content and role.
        """
        # Get last message per conversation
        last_msgs = self.msg_df.sort_values('conversation_id').groupby('conversation_id').last().reset_index()
        return last_msgs[['conversation_id', 'role', 'content']]


    def get_top_phrases(self, n_gram_range=(2, 2), top_k=20) -> List[Tuple[str, int]]:
        """Extracts top n-grams from user messages"""
        if self.user_msgs.empty:
            return []
            
        vectorizer = CountVectorizer(
            stop_words=self.custom_stopwords,
            ngram_range=n_gram_range,
            max_features=top_k
        )
        
        try:
            X = vectorizer.fit_transform(self.user_msgs['content'].dropna())
            counts = X.sum(axis=0).A1
            vocab = vectorizer.get_feature_names_out()
            
            phrases = list(zip(vocab, counts))
            return sorted(phrases, key=lambda x: x[1], reverse=True)
        except ValueError:
            # Handle empty vocabulary or other issues
            return []

    def perform_topic_modeling(self, n_clusters=5) -> pd.DataFrame:
        """
        Performs K-Means clustering on conversations.
        Returns conv_df with a new 'cluster' column and a dictionary of cluster terms.
        """
        if self.conv_df.empty or self.conv_df['content'].str.len().sum() == 0:
             self.conv_df['cluster'] = 0
             return self.conv_df, {}

        tfidf = TfidfVectorizer(
            stop_words=self.custom_stopwords,
            max_features=1000,
            min_df=2
        )
        
        try:
            tfidf_matrix = tfidf.fit_transform(self.conv_df['content'])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.conv_df['cluster'] = kmeans.fit_predict(tfidf_matrix)
            
            # Extract top terms per cluster
            cluster_terms = {}
            feature_names = tfidf.get_feature_names_out()
            ordered_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
            
            for i in range(n_clusters):
                top_terms = [feature_names[ind] for ind in ordered_centroids[i, :5]] # Top 5 terms
                cluster_terms[i] = ", ".join(top_terms)
                
            return self.conv_df, cluster_terms
            
        except ValueError:
            self.conv_df['cluster'] = 0
            return self.conv_df, {0: "Insufficient Data"}

    def analyze_sentiment(self) -> pd.DataFrame:
        """
        Adds sentiment polarity to user messages using keyword-based analysis.
        Better suited for German chat messages than TextBlobDE.
        """
        # Positive and negative German keywords commonly used in chats
        positive_words = [
            'danke', 'super', 'toll', 'klasse', 'perfekt', 'wunderbar', 'genial', 
            'top', 'prima', 'ausgezeichnet', 'fantastisch', 'großartig', 'hilfreich',
            'freundlich', 'gut', 'gerne', 'freue', 'zufrieden', 'glücklich', 'dankeschön',
            'lieb', 'nett', 'cool', 'mega', 'hammer', 'spitze', 'beste', 'optimal',
            '👍', '😊', '🙂', '😀', '❤️', '🎉', '👏', '✅', '💪', '😃'
        ]
        
        negative_words = [
            'schlecht', 'leider', 'problem', 'fehler', 'ärgerlich', 'frustrierend',
            'enttäuscht', 'unzufrieden', 'schlimm', 'furchtbar', 'schrecklich', 
            'nervig', 'langsam', 'kompliziert', 'verwirrend', 'schwierig', 'teuer',
            'nicht funktioniert', 'geht nicht', 'klappt nicht', 'verstehe nicht',
            'hilft nicht', 'immer noch', 'schon wieder', 'trotzdem', 'leider nicht',
            '😞', '😠', '😡', '👎', '😢', '😤', '🙁', '😕'
        ]
        
        def calculate_sentiment(text):
            if pd.isna(text):
                return 0.0
            text_lower = text.lower()
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            total = pos_count + neg_count
            if total == 0:
                return 0.0
            
            # Score from -1 to +1
            return (pos_count - neg_count) / total

        self.user_msgs['sentiment'] = self.user_msgs['content'].apply(calculate_sentiment)
        
        # Average per conversation
        avg_sentiment = self.user_msgs.groupby('conversation_id')['sentiment'].mean().reset_index()
        
        # Merge back to conv_df
        if 'sentiment' in self.conv_df.columns:
            self.conv_df = self.conv_df.drop('sentiment', axis=1)
            
        self.conv_df = self.conv_df.merge(avg_sentiment, on='conversation_id', how='left')
        self.conv_df['sentiment'] = self.conv_df['sentiment'].fillna(0)
        
        return self.conv_df

    def get_wordcloud_text(self) -> str:
        """Returns a single string of all user text for wordcloud"""
        return " ".join(self.user_msgs['content'].dropna().tolist())

    def get_first_questions(self, top_k=15) -> pd.DataFrame:
        """
        Extracts the first user message from each conversation.
        Returns the most common first questions.
        """
        # Get first user message per conversation
        first_user_msgs = self.user_msgs.sort_values('conversation_id').groupby('conversation_id').first().reset_index()
        
        # Count most common first messages
        first_q_counts = first_user_msgs['content'].value_counts().head(top_k).reset_index()
        first_q_counts.columns = ['Frage', 'Anzahl']
        
        return first_q_counts

    def get_bot_helplessness(self) -> Dict:
        """
        Detects bot messages that indicate inability to help.
        Returns statistics, examples, and full list of helpless messages.
        """
        bot_msgs = self.msg_df[self.msg_df['role'] == 'assistant'].copy()
        
        # Keywords that indicate the bot doesn't know or refers to support
        helpless_keywords = [
            'weiß ich nicht', 'kann ich nicht', 'keine information',
            'support kontaktieren', 'kundenservice', 'rufen sie an',
            'leider nicht', 'tut mir leid', 'nicht möglich',
            'wende dich an', 'kontaktiere', 'hilfe vom team'
        ]
        
        # Check which messages contain helpless keywords
        def contains_helpless(text):
            if pd.isna(text):
                return False
            text_lower = text.lower()
            return any(kw in text_lower for kw in helpless_keywords)
        
        bot_msgs['is_helpless'] = bot_msgs['content'].apply(contains_helpless)
        
        helpless_count = bot_msgs['is_helpless'].sum()
        total_bot_msgs = len(bot_msgs)
        helpless_rate = (helpless_count / total_bot_msgs * 100) if total_bot_msgs > 0 else 0
        
        # Get example helpless messages (top 5)
        helpless_examples = bot_msgs[bot_msgs['is_helpless']]['content'].value_counts().head(5).reset_index()
        helpless_examples.columns = ['Nachricht', 'Anzahl']
        
        # Get ALL helpless messages for download
        all_helpless = bot_msgs[bot_msgs['is_helpless']][['conversation_id', 'content']].copy()
        all_helpless.columns = ['Conversation ID', 'Nachricht']
        
        return {
            'helpless_count': int(helpless_count),
            'total_bot_messages': int(total_bot_msgs),
            'helpless_rate': round(helpless_rate, 2),
            'examples': helpless_examples,
            'all_helpless': all_helpless
        }

    def get_response_length_stats(self) -> pd.DataFrame:
        """
        Analyzes the length of bot responses.
        Returns statistics about response lengths.
        """
        bot_msgs = self.msg_df[self.msg_df['role'] == 'assistant'].copy()
        bot_msgs['char_count'] = bot_msgs['content'].str.len()
        bot_msgs['word_count'] = bot_msgs['content'].str.split().str.len()
        
        stats = {
            'avg_chars': round(bot_msgs['char_count'].mean(), 1),
            'avg_words': round(bot_msgs['word_count'].mean(), 1),
            'min_chars': int(bot_msgs['char_count'].min()),
            'max_chars': int(bot_msgs['char_count'].max())
        }
        
        return stats, bot_msgs[['char_count', 'word_count']]
"# Updated" 
