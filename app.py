import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils import load_data
import os

# Page Config
st.set_page_config(page_title="Chatbase Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Dark Mode adjustments if needed (Streamlit handles most)
st.markdown("""
<style>
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Chatbase Analytics")

# File Selection
default_file = "chats_BErjZYVjS1TBgb3431XYY_2025-01-08~2026-01-08.csv"
uploaded_file = st.sidebar.file_uploader("Upload Chat Log (CSV)", type="csv")

analyzer = None

if uploaded_file:
    analyzer = load_data(uploaded_file=uploaded_file)
elif os.path.exists(default_file):
    st.sidebar.info(f"Using default file: {default_file}")
    analyzer = load_data(file_path=default_file)
else:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

if not analyzer or analyzer.conv_df.empty:
    st.error("Could not parse data or data is empty.")
    st.stop()

# Data Filters
st.sidebar.header("Filter")

# Date Filter
min_date = analyzer.conv_df['date'].min()
max_date = analyzer.conv_df['date'].max()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Filter Logic
mask = (analyzer.conv_df['date'].dt.date >= start_date) & (analyzer.conv_df['date'].dt.date <= end_date)
filtered_conv_df = analyzer.conv_df.loc[mask]
filtered_msg_df = analyzer.msg_df[analyzer.msg_df['conversation_id'].isin(filtered_conv_df['conversation_id'])]

# Re-initialize analyzer with filtered data for consistent stats/nlp
# (Optimized: Instead of re-parsing, we could update analyzer state, but creating new instance is cleaner for filtered stats)
# However, analyzer computes heavy stuff on init. 
# Better: Methods in analyzer should accept df as argument or we update attributes.
# Let's update the attributes of a fresh analyzer instance or modify the existing one carefully.
# Modifying the existing one is risky if we want to reset.
# Simpler: Create a new lightweight analyzer or just use filtered DFs for charts, and call analyzer methods on filtered DFs.
# But analyzer methods use self.conv_df. 
# Let's instantiate a new Analyzer with filtered data.
filtered_analyzer = analyzer # Fallback
try:
    # We need to manually construct it to avoid re-downloading stopwords etc if possible, but init is fast enough except downloads.
    from src.analyzer import ChatAnalyzer
    filtered_analyzer = ChatAnalyzer(filtered_conv_df, filtered_msg_df)
except Exception as e:
    st.error(f"Error filtering data: {e}")

# Layout
st.title("🤖 Chatbot Conversation Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Überblick", "💬 Themen & Inhalte", "🧠 Qualität & Sentiment", "📂 Daten-Explorer"])

with tab1:
    st.header("Deskriptive Statistik")
    stats = filtered_analyzer.get_basic_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Konversationen", stats['total_conversations'])
    col2.metric("Nachrichten", stats['total_messages'])
    col3.metric("Ø Dauer (sek)", stats['avg_duration_seconds'])
    col4.metric("Ø Msgs/Conv", stats['avg_messages_per_conversation'])
    
    st.divider()
    
    # Timeline
    st.subheader("Entwicklung über Zeit")
    daily_counts = filtered_analyzer.get_time_distribution(freq='D')
    fig_timeline = px.line(daily_counts, x='date', y='count', title='Tägliche Konversationen', markers=True)
    fig_timeline.update_layout(xaxis_title="Datum", yaxis_title="Anzahl")
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Heatmap (Weekday x Hour)
    st.subheader("Support-Auslastung (Heatmap)")
    heatmap_data = filtered_analyzer.get_heatmap_data()
    if not heatmap_data.empty:
        fig_heatmap = px.imshow(heatmap_data, 
                               labels=dict(x="Stunde", y="Wochentag", color="Anzahl"),
                               x=heatmap_data.columns,
                               y=heatmap_data.index,
                               title="Konversationen nach Zeit & Tag",
                               aspect="auto")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Zu wenig Daten für Heatmap.")

    # Source Distribution if available
    if 'source' in filtered_conv_df.columns and filtered_conv_df['source'].nunique() > 1:
        st.subheader("Quellen")
        source_counts = filtered_conv_df['source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
        fig_source = px.pie(source_counts, values='count', names='source', title='Verteilung nach Quelle')
        st.plotly_chart(fig_source, use_container_width=True)

with tab2:
    st.header("Themen & Inhalte")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Häufigste Phrasen (Bigrams)")
        top_phrases = filtered_analyzer.get_top_phrases(n_gram_range=(2,2), top_k=10)
        if top_phrases:
            df_phrases = pd.DataFrame(top_phrases, columns=['Phrase', 'Count'])
            fig_bar = px.bar(df_phrases, x='Count', y='Phrase', orientation='h', title='Top 10 Bigrams')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Keine Daten für Phrasen verfügbar.")

    with col2:
        st.subheader("Häufigste Phrasen (Trigrams)")
        top_trigrams = filtered_analyzer.get_top_phrases(n_gram_range=(3,3), top_k=10)
        if top_trigrams:
            df_tri = pd.DataFrame(top_trigrams, columns=['Phrase', 'Count'])
            fig_bar_tri = px.bar(df_tri, x='Count', y='Phrase', orientation='h', title='Top 10 Trigrams')
            fig_bar_tri.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar_tri, use_container_width=True)
        else:
            st.info("Keine Daten für Trigrams verfügbar.")
            
    st.divider()
    
    st.subheader("Topic Clustering")
    if st.button("Start Topic Modeling"):
        with st.spinner("Analysiere Themen..."):
            clustered_df, cluster_terms = filtered_analyzer.perform_topic_modeling(n_clusters=5)
            
            # Show stats per cluster
            cluster_stats = clustered_df.groupby('cluster').agg({
                'conversation_id': 'count',
                'duration_seconds': 'mean'
            }).reset_index()
            cluster_stats['Topic Terms'] = cluster_stats['cluster'].map(cluster_terms)
            
            st.dataframe(cluster_stats.style.format({'duration_seconds': '{:.1f}'}))
            
            # Visualize
            fig_cluster = px.bar(cluster_stats, x='cluster', y='conversation_id', 
                                 hover_data=['Topic Terms'], 
                                 title='Konversationen pro Cluster',
                                 labels={'conversation_id': 'Anzahl', 'cluster': 'Cluster ID'})
            st.plotly_chart(fig_cluster, use_container_width=True)
            
    st.divider()
    
    st.subheader("Word Cloud")
    # We can't display matplotlib easily in some envs, but Streamlit supports it.
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    text = filtered_analyzer.get_wordcloud_text()
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='black', 
                              stopwords=filtered_analyzer.custom_stopwords).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("Nicht genügend Text für Wordcloud.")

    st.divider()
    st.subheader("Exit-Analyse (Wie enden Gespräche?)")
    
    exit_df = filtered_analyzer.get_exit_analysis()
    
    # 1. Who sent the last message?
    last_role_counts = exit_df['role'].value_counts().reset_index()
    last_role_counts.columns = ['Role', 'Count']
    fig_exit_role = px.pie(last_role_counts, values='Count', names='Role', 
                          title='Wer hat die letzte Nachricht gesendet?',
                          color='Role', color_discrete_map={'user': 'orange', 'assistant': 'blue'})
    st.plotly_chart(fig_exit_role, use_container_width=True)
    
    # 2. Top last messages (if bot)
    st.markdown("**Häufigste letzte Antworten (Assistant):**")
    bot_exits = exit_df[exit_df['role'] == 'assistant']['content'].value_counts().head(10).reset_index()
    bot_exits.columns = ['Nachricht', 'Anzahl']
    st.dataframe(bot_exits)

with tab3:
    st.header("Qualitäts-Analyse")
    
    if st.button("Sentiment-Analyse durchführen"):
        with st.spinner("Berechne Sentiment..."):
            sentiment_df = filtered_analyzer.analyze_sentiment()
            
            # Histogram
            fig_hist = px.histogram(sentiment_df, x='sentiment', nbins=20, title='Sentiment-Verteilung (Polarity)')
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Scatter Duration vs Sentiment
            fig_scatter = px.scatter(sentiment_df, x='duration_seconds', y='sentiment', 
                                     title='Dauer vs. Sentiment', opacity=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.write("Skala: -1 (Negativ) bis +1 (Positiv)")

    st.divider()
    st.subheader("Komplexitäts-Verteilung (Nachrichtenanzahl)")
    
    # Message Count Distribution
    fig_msg_dist = px.histogram(filtered_conv_df, x='message_count', nbins=30, 
                               title='Verteilung der Nachrichten pro Konversation',
                               labels={'message_count': 'Anzahl Nachrichten'})
    st.plotly_chart(fig_msg_dist, use_container_width=True)
    
    # Classification (Short/Medium/Long)
    def classify_length(n):
        if n <= 3: return "Kurz (1-3)"
        if n <= 10: return "Mittel (4-10)"
        return "Lang (>10)"
    
    filtered_conv_df['length_class'] = filtered_conv_df['message_count'].apply(classify_length)
    class_counts = filtered_conv_df['length_class'].value_counts().reset_index()
    class_counts.columns = ['Klasse', 'Anzahl']
    
    fig_class = px.pie(class_counts, values='Anzahl', names='Klasse', title='Klassifizierung nach Länge')
    st.plotly_chart(fig_class, use_container_width=True)

with tab4:
    st.header("Daten-Explorer")
    search_term = st.text_input("Suche in Nachrichten")
    
    if search_term:
        results = filtered_msg_df[filtered_msg_df['content'].str.contains(search_term, case=False, na=False)]
        st.write(f"{len(results)} Treffer gefunden:")
        st.dataframe(results[['conversation_id', 'role', 'content']])
    else:
        st.subheader("Konversationen")
        st.dataframe(filtered_conv_df)
        st.subheader("Nachrichten")
        st.dataframe(filtered_msg_df.head(100))
