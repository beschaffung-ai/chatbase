import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
import re
import json
from typing import Dict, List, Tuple, Optional


class ChatAnalyzer:
    def __init__(self, conv_df: pd.DataFrame, msg_df: pd.DataFrame):
        self.conv_df = conv_df.copy() if conv_df is not None else pd.DataFrame()
        self.msg_df = msg_df.copy() if msg_df is not None else pd.DataFrame()
        self._prepare_data()

    def _prepare_data(self):
        # Validate and fix DataFrames
        self._validate_dataframes()

        # Filter for user messages for content analysis
        if "role" in self.msg_df.columns:
            self.user_msgs = self.msg_df[self.msg_df["role"] == "user"].copy()
        else:
            self.user_msgs = self.msg_df.copy()

        # Ensure dates are datetime
        if "date" in self.conv_df.columns:
            self.conv_df["date"] = pd.to_datetime(self.conv_df["date"], errors="coerce")
            # Fill NaT with current time
            self.conv_df["date"] = self.conv_df["date"].fillna(pd.Timestamp.now())
        else:
            self.conv_df["date"] = pd.Timestamp.now()

        # Clean up potentially existing columns from previous runs/filtering
        for col in ["content", "cluster", "sentiment"]:
            if col in self.conv_df.columns:
                self.conv_df = self.conv_df.drop(col, axis=1)

        # Aggregate user text per conversation for clustering
        if not self.user_msgs.empty and "content" in self.user_msgs.columns:
            self.conv_text = (
                self.user_msgs.groupby("conversation_id")["content"]
                .apply(lambda x: " ".join(str(v) for v in x if pd.notna(v)))
                .reset_index()
            )
            self.conv_df = self.conv_df.merge(
                self.conv_text, on="conversation_id", how="left"
            )
        else:
            self.conv_df["content"] = ""

        self.conv_df["content"] = self.conv_df["content"].fillna("")

        # Download nltk resources if needed
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        try:
            self.german_stopwords = stopwords.words("german")
        except:
            self.german_stopwords = []

        # Add custom stopwords relevant to chat context
        self.custom_stopwords = self.german_stopwords + [
            "hallo",
            "hi",
            "danke",
            "bitte",
            "ja",
            "nein",
            "ok",
            "okay",
            "guten",
            "tag",
        ]

    def _validate_dataframes(self):
        """Ensure DataFrames have required columns with proper types."""
        # Required columns for conv_df
        conv_required = {
            "conversation_id": "object",
            "date": "datetime64[ns]",
            "duration_seconds": "float64",
            "message_count": "int64",
        }

        # Required columns for msg_df
        msg_required = {
            "conversation_id": "object",
            "role": "object",
            "content": "object",
        }

        # Add missing columns with defaults
        for col, dtype in conv_required.items():
            if col not in self.conv_df.columns:
                if dtype == "datetime64[ns]":
                    self.conv_df[col] = pd.Timestamp.now()
                elif dtype == "float64":
                    self.conv_df[col] = 0.0
                elif dtype == "int64":
                    self.conv_df[col] = 0
                else:
                    self.conv_df[col] = ""

        for col, dtype in msg_required.items():
            if col not in self.msg_df.columns:
                self.msg_df[col] = ""

        # Ensure content is string
        if "content" in self.msg_df.columns:
            self.msg_df["content"] = self.msg_df["content"].astype(str)

    def get_basic_stats(self) -> Dict:
        """Returns descriptive statistics"""
        total_conv = len(self.conv_df)
        total_msgs = len(self.msg_df)
        avg_duration = self.conv_df["duration_seconds"].mean()
        avg_msgs_per_conv = self.msg_df.groupby("conversation_id").size().mean()

        return {
            "total_conversations": total_conv,
            "total_messages": total_msgs,
            "avg_duration_seconds": (
                round(avg_duration, 2) if not np.isnan(avg_duration) else 0
            ),
            "avg_messages_per_conversation": round(avg_msgs_per_conv, 2),
        }

    def get_time_distribution(self, freq="D") -> pd.DataFrame:
        """Returns conversation counts aggregated by time frequency (D=Daily, W=Weekly, M=Monthly)"""
        return (
            self.conv_df.set_index("date")
            .resample(freq)
            .size()
            .reset_index(name="count")
        )

    def get_heatmap_data(self) -> pd.DataFrame:
        """Returns a pivot table for Weekday vs Hour heatmap"""
        df = self.conv_df.copy()
        df["weekday"] = df["date"].dt.day_name()
        # Order weekdays
        cats = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        df["weekday"] = pd.Categorical(df["weekday"], categories=cats, ordered=True)
        df["hour"] = df["date"].dt.hour

        heatmap_data = df.groupby(["weekday", "hour"]).size().reset_index(name="count")
        return heatmap_data.pivot(
            index="weekday", columns="hour", values="count"
        ).fillna(0)

    def get_exit_analysis(self) -> pd.DataFrame:
        """
        Analyzes the last message of each conversation.
        Returns a DataFrame with last message content and role.
        """
        # Get last message per conversation
        last_msgs = (
            self.msg_df.sort_values("conversation_id")
            .groupby("conversation_id")
            .last()
            .reset_index()
        )
        return last_msgs[["conversation_id", "role", "content"]]

    def get_top_phrases(self, n_gram_range=(2, 2), top_k=20) -> List[Tuple[str, int]]:
        """Extracts top n-grams from user messages"""
        if self.user_msgs.empty:
            return []

        vectorizer = CountVectorizer(
            stop_words=self.custom_stopwords,
            ngram_range=n_gram_range,
            max_features=top_k,
        )

        try:
            X = vectorizer.fit_transform(self.user_msgs["content"].dropna())
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
        if self.conv_df.empty or self.conv_df["content"].str.len().sum() == 0:
            self.conv_df["cluster"] = 0
            return self.conv_df, {}

        tfidf = TfidfVectorizer(
            stop_words=self.custom_stopwords, max_features=1000, min_df=2
        )

        try:
            tfidf_matrix = tfidf.fit_transform(self.conv_df["content"])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.conv_df["cluster"] = kmeans.fit_predict(tfidf_matrix)

            # Extract top terms per cluster
            cluster_terms = {}
            feature_names = tfidf.get_feature_names_out()
            ordered_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

            for i in range(n_clusters):
                top_terms = [
                    feature_names[ind] for ind in ordered_centroids[i, :5]
                ]  # Top 5 terms
                cluster_terms[i] = ", ".join(top_terms)

            return self.conv_df, cluster_terms

        except ValueError:
            self.conv_df["cluster"] = 0
            return self.conv_df, {0: "Insufficient Data"}

    def analyze_sentiment(self) -> pd.DataFrame:
        """
        Adds sentiment polarity to user messages using keyword-based analysis.
        Better suited for German chat messages than TextBlobDE.
        """
        # Positive and negative German keywords commonly used in chats
        positive_words = [
            "danke",
            "super",
            "toll",
            "klasse",
            "perfekt",
            "wunderbar",
            "genial",
            "top",
            "prima",
            "ausgezeichnet",
            "fantastisch",
            "großartig",
            "hilfreich",
            "freundlich",
            "gut",
            "gerne",
            "freue",
            "zufrieden",
            "glücklich",
            "dankeschön",
            "lieb",
            "nett",
            "cool",
            "mega",
            "hammer",
            "spitze",
            "beste",
            "optimal",
            "👍",
            "😊",
            "🙂",
            "😀",
            "❤️",
            "🎉",
            "👏",
            "✅",
            "💪",
            "😃",
        ]

        negative_words = [
            "schlecht",
            "leider",
            "problem",
            "fehler",
            "ärgerlich",
            "frustrierend",
            "enttäuscht",
            "unzufrieden",
            "schlimm",
            "furchtbar",
            "schrecklich",
            "nervig",
            "langsam",
            "kompliziert",
            "verwirrend",
            "schwierig",
            "teuer",
            "nicht funktioniert",
            "geht nicht",
            "klappt nicht",
            "verstehe nicht",
            "hilft nicht",
            "immer noch",
            "schon wieder",
            "trotzdem",
            "leider nicht",
            "😞",
            "😠",
            "😡",
            "👎",
            "😢",
            "😤",
            "🙁",
            "😕",
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

        self.user_msgs["sentiment"] = self.user_msgs["content"].apply(
            calculate_sentiment
        )

        # Average per conversation
        avg_sentiment = (
            self.user_msgs.groupby("conversation_id")["sentiment"].mean().reset_index()
        )

        # Merge back to conv_df
        if "sentiment" in self.conv_df.columns:
            self.conv_df = self.conv_df.drop("sentiment", axis=1)

        self.conv_df = self.conv_df.merge(
            avg_sentiment, on="conversation_id", how="left"
        )
        self.conv_df["sentiment"] = self.conv_df["sentiment"].fillna(0)

        return self.conv_df

    def get_wordcloud_text(self) -> str:
        """Returns a single string of all user text for wordcloud"""
        return " ".join(self.user_msgs["content"].dropna().tolist())

    def get_first_questions(self, top_k=15) -> pd.DataFrame:
        """
        Extracts the first user message from each conversation.
        Returns the most common first questions.
        """
        # Get first user message per conversation
        first_user_msgs = (
            self.user_msgs.sort_values("conversation_id")
            .groupby("conversation_id")
            .first()
            .reset_index()
        )

        # Count most common first messages
        first_q_counts = (
            first_user_msgs["content"].value_counts().head(top_k).reset_index()
        )
        first_q_counts.columns = ["Frage", "Anzahl"]

        return first_q_counts

    def get_bot_helplessness(self) -> Dict:
        """
        Detects bot messages that indicate inability to help.
        Returns statistics, examples, and full list of helpless messages.
        """
        bot_msgs = self.msg_df[self.msg_df["role"] == "assistant"].copy()

        # Keywords that indicate the bot doesn't know or refers to support
        helpless_keywords = [
            "weiß ich nicht",
            "kann ich nicht",
            "keine information",
            "support kontaktieren",
            "kundenservice",
            "rufen sie an",
            "leider nicht",
            "tut mir leid",
            "nicht möglich",
            "wende dich an",
            "kontaktiere",
            "hilfe vom team",
        ]

        # Check which messages contain helpless keywords
        def contains_helpless(text):
            if pd.isna(text):
                return False
            text_lower = text.lower()
            return any(kw in text_lower for kw in helpless_keywords)

        bot_msgs["is_helpless"] = bot_msgs["content"].apply(contains_helpless)

        helpless_count = bot_msgs["is_helpless"].sum()
        total_bot_msgs = len(bot_msgs)
        helpless_rate = (
            (helpless_count / total_bot_msgs * 100) if total_bot_msgs > 0 else 0
        )

        # Get example helpless messages (top 5)
        helpless_examples = (
            bot_msgs[bot_msgs["is_helpless"]]["content"]
            .value_counts()
            .head(5)
            .reset_index()
        )
        helpless_examples.columns = ["Nachricht", "Anzahl"]

        # Get ALL helpless messages for download
        all_helpless = bot_msgs[bot_msgs["is_helpless"]][
            ["conversation_id", "content"]
        ].copy()
        all_helpless.columns = ["Conversation ID", "Nachricht"]

        return {
            "helpless_count": int(helpless_count),
            "total_bot_messages": int(total_bot_msgs),
            "helpless_rate": round(helpless_rate, 2),
            "examples": helpless_examples,
            "all_helpless": all_helpless,
        }

    def get_response_length_stats(self) -> pd.DataFrame:
        """
        Analyzes the length of bot responses.
        Returns statistics about response lengths.
        """
        bot_msgs = self.msg_df[self.msg_df["role"] == "assistant"].copy()
        bot_msgs["char_count"] = bot_msgs["content"].str.len()
        bot_msgs["word_count"] = bot_msgs["content"].str.split().str.len()

        stats = {
            "avg_chars": round(bot_msgs["char_count"].mean(), 1),
            "avg_words": round(bot_msgs["word_count"].mean(), 1),
            "min_chars": int(bot_msgs["char_count"].min()),
            "max_chars": int(bot_msgs["char_count"].max()),
        }

        return stats, bot_msgs[["char_count", "word_count"]]

    def calculate_success_rate(self) -> Dict:
        """
        Calculates conversation success rate based on last message analysis.
        - Success: User ends with positive keywords (danke, super, perfekt, etc.)
        - Failure: Bot ends with helpless keywords OR user ends with negative keywords
        - Neutral: Everything else
        """
        positive_endings = [
            "danke",
            "super",
            "perfekt",
            "hat geholfen",
            "wunderbar",
            "toll",
            "funktioniert",
            "geklappt",
            "klasse",
            "top",
            "prima",
            "genial",
            "vielen dank",
            "dankeschön",
            "ausgezeichnet",
            "sehr gut",
            "hilfreich",
            "👍",
            "😊",
            "🙂",
            "❤️",
            "🎉",
            "👏",
            "✅",
        ]

        negative_endings = [
            "hilft nicht",
            "verstehe nicht",
            "funktioniert nicht",
            "geht nicht",
            "klappt nicht",
            "schlecht",
            "unzufrieden",
            "enttäuscht",
            "frustrierend",
            "ärgerlich",
            "nervig",
            "immer noch nicht",
            "schon wieder",
            "😞",
            "😠",
            "😡",
            "👎",
            "😢",
            "😤",
        ]

        helpless_keywords = [
            "weiß ich nicht",
            "kann ich nicht",
            "keine information",
            "support kontaktieren",
            "kundenservice",
            "rufen sie an",
            "leider nicht möglich",
            "nicht möglich",
            "wende dich an",
        ]

        # Get last message per conversation
        last_msgs = (
            self.msg_df.sort_values("conversation_id")
            .groupby("conversation_id")
            .last()
            .reset_index()
        )

        def classify_ending(row):
            if pd.isna(row["content"]):
                return "neutral"
            text_lower = row["content"].lower()

            # User ended with positive = Success
            if row["role"] == "user":
                if any(kw in text_lower for kw in positive_endings):
                    return "success"
                if any(kw in text_lower for kw in negative_endings):
                    return "failure"

            # Bot ended with helpless keywords = Failure
            if row["role"] == "assistant":
                if any(kw in text_lower for kw in helpless_keywords):
                    return "failure"

            return "neutral"

        last_msgs["outcome"] = last_msgs.apply(classify_ending, axis=1)

        outcome_counts = last_msgs["outcome"].value_counts().to_dict()
        total = len(last_msgs)

        success_count = outcome_counts.get("success", 0)
        failure_count = outcome_counts.get("failure", 0)
        neutral_count = outcome_counts.get("neutral", 0)

        success_rate = (success_count / total * 100) if total > 0 else 0

        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "neutral_count": neutral_count,
            "total": total,
            "success_rate": round(success_rate, 1),
            "outcome_df": last_msgs[["conversation_id", "role", "content", "outcome"]],
        }

    def get_keyword_trends(self, freq="W", top_k=10) -> pd.DataFrame:
        """
        Extracts top keywords over time.
        Returns a DataFrame with date, keyword, and count for trend visualization.
        """
        # Merge user messages with dates from conversations
        user_with_dates = self.user_msgs.merge(
            self.conv_df[["conversation_id", "date"]], on="conversation_id", how="left"
        )

        # Resample by frequency
        user_with_dates["period"] = user_with_dates["date"].dt.to_period(freq)

        # Get unique periods
        periods = user_with_dates["period"].dropna().unique()

        trend_data = []

        for period in periods:
            period_msgs = user_with_dates[user_with_dates["period"] == period]
            text = " ".join(period_msgs["content"].dropna())

            if len(text.strip()) < 10:
                continue

            # Extract keywords using CountVectorizer
            try:
                vectorizer = CountVectorizer(
                    stop_words=self.custom_stopwords,
                    ngram_range=(1, 1),
                    max_features=50,
                )
                X = vectorizer.fit_transform([text])
                counts = X.sum(axis=0).A1
                vocab = vectorizer.get_feature_names_out()

                for word, count in zip(vocab, counts):
                    trend_data.append(
                        {
                            "period": period.to_timestamp(),
                            "keyword": word,
                            "count": int(count),
                        }
                    )
            except ValueError:
                continue

        if not trend_data:
            return pd.DataFrame(columns=["period", "keyword", "count"])

        trend_df = pd.DataFrame(trend_data)

        # Get overall top keywords
        top_keywords = (
            trend_df.groupby("keyword")["count"].sum().nlargest(top_k).index.tolist()
        )

        # Filter to only top keywords
        trend_df = trend_df[trend_df["keyword"].isin(top_keywords)]

        return trend_df

    def perform_ai_topic_modeling(
        self, sample_size: int = 500, model: str = "gpt-4o"
    ) -> Dict:
        """
        Uses OpenAI GPT-5.1 to extract topics from conversations.
        More accurate than K-Means for semantic understanding.

        GPT-5.1 Features:
        - 400K context window (allows sending many more conversations)
        - Better reasoning for topic extraction
        - Structured JSON output

        Args:
            sample_size: Number of conversations to sample (default 500 for better coverage)

        Returns:
            Dictionary with topics and metadata
        """
        import streamlit as st

        try:
            import openai
            from openai import OpenAI
        except ImportError:
            return {"error": "OpenAI package not installed", "topics": []}

        # Check if API key is configured
        try:
            api_key = st.secrets["openai"]["api_key"]
        except Exception:
            return {"error": "OpenAI API key not configured in secrets", "topics": []}

        # Initialize client
        client = OpenAI(api_key=api_key)

        # ===== STRATIFIED SAMPLING =====
        # Ensure we get a representative sample across time and conversation lengths
        df = self.conv_df.copy()
        actual_sample_size = min(sample_size, len(df))

        try:
            # Calculate conversation length quartiles for stratification
            if "message_count" in df.columns and df["message_count"].nunique() > 4:
                df["length_bucket"] = pd.qcut(
                    df["message_count"], q=4, duplicates="drop"
                )
            else:
                df["length_bucket"] = "all"

            # Calculate time quartiles for stratification
            if "date" in df.columns and df["date"].nunique() > 4:
                df["time_bucket"] = pd.qcut(
                    df["date"].astype(int), q=4, duplicates="drop"
                )
            else:
                df["time_bucket"] = "all"

            # Stratified sampling: try to get equal samples from each bucket
            sampled_dfs = []
            length_buckets = df["length_bucket"].unique()
            time_buckets = df["time_bucket"].unique()
            n_buckets = len(length_buckets) * len(time_buckets)

            if n_buckets > 1:
                for length in length_buckets:
                    for time in time_buckets:
                        bucket_df = df[
                            (df["length_bucket"] == length)
                            & (df["time_bucket"] == time)
                        ]
                        if not bucket_df.empty:
                            n_samples = max(1, actual_sample_size // n_buckets)
                            sampled_dfs.append(
                                bucket_df.sample(min(n_samples, len(bucket_df)))
                            )

                sample_df = (
                    pd.concat(sampled_dfs)
                    if sampled_dfs
                    else df.sample(actual_sample_size)
                )
            else:
                sample_df = df.sample(actual_sample_size)

        except Exception:
            # Fallback to random sampling if stratification fails
            sample_df = df.sample(actual_sample_size)

        # ===== PREPARE CONVERSATION TEXTS =====
        # With 400K context window, we can send more text per conversation
        conv_texts = []
        for _, row in sample_df.iterrows():
            text = str(row.get("content", ""))[
                :2000
            ]  # 2000 chars per conversation (4x more than before)
            if text.strip():
                # Add metadata for context
                msg_count = row.get("message_count", "unknown")
                conv_texts.append(f"[{msg_count} Nachrichten] {text}")

        if not conv_texts:
            return {"error": "No conversation texts found", "topics": []}

        # With 400K context, we can send up to ~300 conversations safely
        max_convs = min(300, len(conv_texts))
        combined = "\n\n---\n\n".join(conv_texts[:max_convs])

        try:
            response = client.chat.completions.create(
                model=model,  # User can choose: gpt-4o, gpt-4o-mini, gpt-5.1, etc.
                messages=[
                    {
                        "role": "system",
                        "content": """Du bist ein Senior Data Analyst spezialisiert auf Chatbot-Konversationsanalyse. 

Deine Aufgabe:
1. Analysiere die folgenden Kundenanfragen aus einem Chatbot
2. Identifiziere die HAUPTTHEMEN (5-15 Themen je nach Diversität)
3. Erkenne Muster und Trends
4. Bewerte die Häufigkeit jedes Themas

Wichtig:
- Fasse ähnliche Themen zusammen (z.B. "Lieferung" und "Versand" = ein Thema)
- Erkenne auch Unterprobleme (z.B. "Retoure" hat Unterthemen wie "Rücksendeschein", "Erstattung")
- Identifiziere emotionale Muster (Frustration, Zufriedenheit)

Antworte NUR im JSON-Format:
{
    "topics": [
        {
            "name": "Themenname",
            "description": "Detaillierte Beschreibung was Kunden zu diesem Thema fragen",
            "frequency": "hoch/mittel/niedrig",
            "estimated_percentage": 15,
            "example_keywords": ["keyword1", "keyword2", "keyword3"],
            "subtopics": ["Unterthema1", "Unterthema2"],
            "sentiment_tendency": "positiv/neutral/negativ/gemischt"
        }
    ],
    "summary": "Executive Summary der Hauptanliegen (2-3 Sätze)",
    "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
    "recommendations": ["Empfehlung 1", "Empfehlung 2"]
}""",
                    },
                    {
                        "role": "user",
                        "content": f"Analysiere diese {len(conv_texts[:max_convs])} Kundenanfragen aus einem Chatbot. Die Stichprobe wurde stratifiziert nach Gesprächslänge und Zeitraum ausgewählt, um repräsentativ zu sein:\n\n{combined}",
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=4000,  # Maximum output tokens for detailed analysis
                temperature=0.2,  # Lower temperature for more consistent analysis
            )

            # Parse response
            response_content = response.choices[0].message.content

            if not response_content:
                return {"error": "OpenAI returned empty response", "topics": []}

            try:
                result = json.loads(response_content)
            except json.JSONDecodeError as je:
                return {
                    "error": f"Invalid JSON response: {str(je)[:100]}",
                    "raw_response": response_content[:500],
                    "topics": [],
                }

            result["conversations_analyzed"] = len(conv_texts[:max_convs])
            result["total_conversations"] = len(self.conv_df)
            result["sample_coverage"] = round(
                len(conv_texts[:max_convs]) / len(self.conv_df) * 100, 1
            )
            result["model_used"] = model
            result["sampling_method"] = "stratified (time + length)"

            return result

        except openai.APIError as e:
            return {"error": f"OpenAI API Error: {e.message}", "topics": []}
        except openai.APIConnectionError as e:
            return {"error": f"Connection Error: Could not reach OpenAI", "topics": []}
        except openai.RateLimitError as e:
            return {
                "error": f"Rate Limit: Too many requests, please wait",
                "topics": [],
            }
        except openai.AuthenticationError as e:
            return {"error": f"Authentication Error: Invalid API key", "topics": []}
        except Exception as e:
            return {
                "error": f"Unexpected error: {type(e).__name__}: {str(e)}",
                "topics": [],
            }

    def perform_embedding_clustering(
        self,
        n_clusters: int = 8,
        use_hdbscan: bool = True,
        min_cluster_size: int = None,
        min_samples: int = 3,
        epsilon: float = 0.0,
        cached_embeddings: List = None,
        progress_callback=None,
    ) -> Dict:
        """
        Performs topic clustering on ALL conversations using OpenAI Embeddings.

        This method:
        1. Creates embeddings for ALL conversations (not just a sample) - OR uses cached ones
        2. Uses UMAP for dimensionality reduction
        3. Clusters with HDBSCAN (auto) or KMeans (fixed n_clusters)
        4. Extracts TF-IDF keywords per cluster
        5. Uses GPT-4o-mini to generate cluster names and descriptions

        Args:
            n_clusters: Number of clusters for KMeans (ignored if use_hdbscan=True)
            use_hdbscan: If True, use HDBSCAN (auto cluster count). If False, use KMeans.
            min_cluster_size: Minimum size for a cluster (HDBSCAN only). If None, calculated dynamically.
            min_samples: Measure of how conservative the clustering is (HDBSCAN only).
            epsilon: Cluster selection epsilon (HDBSCAN only).
            cached_embeddings: Optional list of pre-computed embeddings to avoid API calls.
            progress_callback: Optional callable for progress updates (0.0 to 1.0)

        Returns:
            Dictionary with clustered data, descriptions, visualization data, and raw embeddings.
        """
        import streamlit as st

        # ===== IMPORTS =====
        try:
            from openai import OpenAI
            import umap
            import hdbscan
        except ImportError as e:
            return {
                "error": f"Missing dependency: {e}. Install with: pip install openai umap-learn hdbscan"
            }

        # ===== CHECK API KEY =====
        try:
            api_key = st.secrets["openai"]["api_key"]
        except Exception:
            return {"error": "OpenAI API key not configured in secrets"}

        client = OpenAI(api_key=api_key)

        # ===== PREPARE TEXTS =====
        df = self.conv_df.copy()
        texts = df["content"].fillna("").tolist()

        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 10]
        valid_texts = [
            texts[i][:2000] for i in valid_indices
        ]  # Limit to 2000 chars each

        if len(valid_texts) < 10:
            return {
                "error": "Not enough valid conversations for clustering (minimum 10)"
            }

        if progress_callback:
            progress_callback(0.1, "Texte vorbereitet...")

        # ===== GET EMBEDDINGS =====
        all_embeddings = []

        if cached_embeddings and len(cached_embeddings) == len(valid_texts):
            if progress_callback:
                progress_callback(0.2, "Nutze gespeicherte Embeddings...")
            all_embeddings = cached_embeddings
        else:
            # Create new embeddings
            batch_size = 500  # OpenAI allows up to 2048, but 500 is safer for memory
            total_batches = (len(valid_texts) + batch_size - 1) // batch_size

            try:
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(valid_texts))
                    batch = valid_texts[start_idx:end_idx]

                    response = client.embeddings.create(
                        model="text-embedding-3-large", input=batch
                    )

                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)

                    if progress_callback:
                        progress = 0.1 + 0.5 * (batch_idx + 1) / total_batches
                        progress_callback(
                            progress,
                            f"Embeddings erstellt ({end_idx}/{len(valid_texts)})...",
                        )

            except Exception as e:
                return {"error": f"Embedding API error: {str(e)}"}

        # ===== DIMENSIONALITY REDUCTION WITH UMAP =====
        if progress_callback:
            progress_callback(0.65, "UMAP Dimensionsreduktion...")

        try:
            X = np.array(all_embeddings)

            # UMAP for visualization (2D)
            umap_2d = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                metric="cosine",
                random_state=42,
            )
            embedding_2d = umap_2d.fit_transform(X)

            # UMAP for clustering (higher dimension for better clustering)
            umap_cluster = umap.UMAP(
                n_neighbors=15,
                min_dist=0.0,
                n_components=10,
                metric="cosine",
                random_state=42,
            )
            embedding_cluster = umap_cluster.fit_transform(X)

        except Exception as e:
            return {"error": f"UMAP error: {str(e)}"}

        # ===== CLUSTERING =====
        if progress_callback:
            progress_callback(0.75, "Clustering...")

        try:
            if use_hdbscan:
                # Calculate default if not provided
                if min_cluster_size is None:
                    min_cluster_size = max(5, len(valid_texts) // 50)

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    cluster_selection_epsilon=float(epsilon),
                    metric="euclidean",
                )
                labels = clusterer.fit_predict(embedding_cluster)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embedding_cluster)

        except Exception as e:
            return {"error": f"Clustering error: {str(e)}"}

        # ===== CREATE RESULT DATAFRAME =====
        # Map back to original indices
        result_df = df.iloc[valid_indices].copy()
        result_df["cluster"] = labels
        result_df["umap_x"] = embedding_2d[:, 0]
        result_df["umap_y"] = embedding_2d[:, 1]

        # ===== EXTRACT KEYWORDS PER CLUSTER =====
        if progress_callback:
            progress_callback(0.8, "Keywords extrahieren...")

        unique_clusters = sorted(
            [c for c in set(labels) if c >= 0]
        )  # Exclude noise (-1)
        cluster_info = {}

        for cluster_id in unique_clusters:
            cluster_texts = result_df[result_df["cluster"] == cluster_id][
                "content"
            ].tolist()
            cluster_size = len(cluster_texts)

            # TF-IDF for keywords
            try:
                tfidf = TfidfVectorizer(
                    stop_words=self.custom_stopwords,
                    max_features=20,
                    ngram_range=(1, 2),
                )
                tfidf_matrix = tfidf.fit_transform(cluster_texts)
                feature_names = tfidf.get_feature_names_out()

                # Get top keywords by TF-IDF score sum
                scores = tfidf_matrix.sum(axis=0).A1
                top_indices = scores.argsort()[::-1][:10]
                keywords = [feature_names[i] for i in top_indices]
            except:
                keywords = []

            # Get representative examples
            examples = cluster_texts[:3] if len(cluster_texts) >= 3 else cluster_texts
            examples = [ex[:200] + "..." if len(ex) > 200 else ex for ex in examples]

            cluster_info[cluster_id] = {
                "size": cluster_size,
                "percentage": round(cluster_size / len(valid_texts) * 100, 1),
                "keywords": keywords,
                "examples": examples,
            }

        # Handle noise points
        noise_count = sum(1 for l in labels if l == -1)
        if noise_count > 0:
            cluster_info[-1] = {
                "size": noise_count,
                "percentage": round(noise_count / len(valid_texts) * 100, 1),
                "keywords": ["Sonstige", "Nicht zugeordnet"],
                "examples": [],
                "is_noise": True,
            }

        # ===== GPT CLUSTER NAMING =====
        if progress_callback:
            progress_callback(0.85, "GPT Cluster-Beschreibungen...")

        cluster_descriptions = {}

        # Only describe top clusters (exclude noise and very small clusters)
        clusters_to_describe = [
            c for c in unique_clusters if cluster_info[c]["size"] >= 5
        ][:15]

        for cluster_id in clusters_to_describe:
            info = cluster_info[cluster_id]
            keywords_str = ", ".join(info["keywords"][:8])
            examples_str = "\n".join([f"- {ex}" for ex in info["examples"]])

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Analysiere diese Cluster-Daten und gib einen passenden deutschen Namen und eine kurze Beschreibung.

Keywords: {keywords_str}

Beispiel-Anfragen:
{examples_str}

Antworte im JSON-Format:
{{"name": "Kurzer Themenname (2-4 Wörter)", "description": "Eine Beschreibung in 1-2 Sätzen was Kunden hier fragen", "sentiment": "positiv/neutral/negativ"}}""",
                        }
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=150,
                    temperature=0.3,
                )

                desc = json.loads(response.choices[0].message.content)
                cluster_descriptions[cluster_id] = desc

            except Exception as e:
                cluster_descriptions[cluster_id] = {
                    "name": f"Thema {cluster_id + 1}",
                    "description": f"Cluster mit Keywords: {keywords_str}",
                    "sentiment": "neutral",
                }

        if progress_callback:
            progress_callback(1.0, "Fertig!")

        # ===== PREPARE FINAL RESULT =====
        return {
            "success": True,
            "clustered_df": result_df,
            "cluster_info": cluster_info,
            "cluster_descriptions": cluster_descriptions,
            "total_analyzed": len(valid_texts),
            "total_conversations": len(df),
            "n_clusters": len(unique_clusters),
            "noise_count": noise_count,
            "method": "HDBSCAN" if use_hdbscan else f"KMeans (k={n_clusters})",
            "embedding_model": "text-embedding-3-large",
            "cached_embeddings": all_embeddings,  # Return embeddings for caching
            "visualization_data": {
                "x": embedding_2d[:, 0].tolist(),
                "y": embedding_2d[:, 1].tolist(),
                "labels": labels.tolist(),
                "texts": [t[:100] for t in valid_texts],  # Truncated for hover
            },
        }
