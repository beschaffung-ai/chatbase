import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.utils import load_data
import os

# Page Config
st.set_page_config(page_title="Chatbase Analytics", layout="wide", initial_sidebar_state="expanded")

# ==================== PASSWORD AUTHENTICATION ====================
def check_password():
    """Returns `True` if the user has entered correct credentials."""
    
    # Return True if already logged in
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    st.markdown("### 🔐 Login erforderlich")
    st.markdown("Diese App ist passwortgeschützt.")
    
    # Use session state to store input values
    if "login_username" not in st.session_state:
        st.session_state.login_username = ""
    if "login_password" not in st.session_state:
        st.session_state.login_password = ""
    
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Passwort", type="password", key="login_password")
    
    if st.button("Anmelden"):
        try:
            if (username in st.secrets["passwords"] and 
                st.secrets["passwords"][username] == password):
                st.session_state["password_correct"] = True
                st.session_state["current_user"] = username
                # Clear login fields
                del st.session_state.login_username
                del st.session_state.login_password
                st.rerun()
            else:
                st.error("😕 Falscher Username oder Passwort")
        except Exception as e:
            st.error(f"😕 Fehler: Secrets nicht konfiguriert. Erstelle .streamlit/secrets.toml")
    
    return False

# Check authentication
if not check_password():
    st.stop()

# ==================== END AUTHENTICATION ====================

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
st.sidebar.caption(f"Angemeldet als: {st.session_state.get('current_user', 'Unknown')}")

# File Selection
uploaded_file = st.sidebar.file_uploader("Upload Chat Log (CSV)", type="csv", 
                                         help="Lade hier deinen Chatbase-Export hoch (CSV-Format).")

analyzer = None

if uploaded_file:
    analyzer = load_data(uploaded_file=uploaded_file)
else:
    st.info("👋 Willkommen! Bitte lade eine Chatbase-CSV-Datei hoch, um die Analyse zu starten.")
    st.stop()


if not analyzer or analyzer.conv_df.empty:
    st.error("Could not parse data or data is empty.")
    st.stop()

# Data Filters
st.sidebar.header("Filter", help="Grenze die Analyse auf einen bestimmten Zeitraum ein.")

# Date Filter
min_date = analyzer.conv_df['date'].min()
max_date = analyzer.conv_df['date'].max()

start_date = st.sidebar.date_input("Start Date", min_date, help="Analysiere ab diesem Datum.")
end_date = st.sidebar.date_input("End Date", max_date, help="Analysiere bis zu diesem Datum.")

# HTML Export Section
st.sidebar.divider()
st.sidebar.header("📥 Export", help="Exportiere die Analyse-Ergebnisse als HTML-Datei.")

def generate_html_report(analyzer_obj, start_d, end_d):
    """Generates a comprehensive HTML report with all KPIs and charts."""
    stats = analyzer_obj.get_basic_stats()
    success = analyzer_obj.calculate_success_rate()
    helpless = analyzer_obj.get_bot_helplessness()
    time_dist = analyzer_obj.get_time_distribution(freq='W')
    first_qs = analyzer_obj.get_first_questions(top_k=10)
    top_bigrams = analyzer_obj.get_top_phrases(n_gram_range=(2,2), top_k=10)
    
    # Create charts
    fig_timeline = px.line(time_dist, x='date', y='count', title='Wöchentliche Konversationen', markers=True)
    fig_timeline.update_layout(xaxis_title="Datum", yaxis_title="Anzahl")
    
    outcome_df = pd.DataFrame({
        'Outcome': ['Erfolg', 'Neutral', 'Misserfolg'],
        'Count': [success['success_count'], success['neutral_count'], success['failure_count']]
    })
    fig_success = px.pie(outcome_df, values='Count', names='Outcome', title='Gesprächs-Outcomes',
                         color='Outcome', color_discrete_map={'Erfolg': '#2ECC71', 'Neutral': '#95A5A6', 'Misserfolg': '#E74C3C'})
    
    if top_bigrams:
        df_bigrams = pd.DataFrame(top_bigrams, columns=['Phrase', 'Count'])
        fig_bigrams = px.bar(df_bigrams, x='Count', y='Phrase', orientation='h', title='Top 10 Bigrams')
        fig_bigrams.update_layout(yaxis={'categoryorder':'total ascending'})
        bigrams_html = fig_bigrams.to_html(full_html=False, include_plotlyjs=False)
    else:
        bigrams_html = "<p>Keine Daten für Bigrams.</p>"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbase Analytics Report</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #1a1a2e;
                color: #eaeaea;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2 {{
                color: #00d4ff;
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .kpi-card {{
                background: #16213e;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            }}
            .kpi-value {{
                font-size: 2em;
                font-weight: bold;
                color: #00d4ff;
            }}
            .kpi-label {{
                color: #888;
                margin-top: 5px;
            }}
            .chart-container {{
                background: #16213e;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
            }}
            .section {{
                margin: 40px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #333;
            }}
            th {{
                background: #0f3460;
            }}
            .success {{ color: #2ECC71; }}
            .neutral {{ color: #95A5A6; }}
            .failure {{ color: #E74C3C; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Chatbase Analytics Report</h1>
            <p>Zeitraum: {start_d} bis {end_d}</p>
            
            <div class="section">
                <h2>📈 Übersicht</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['total_conversations']}</div>
                        <div class="kpi-label">Konversationen</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['total_messages']}</div>
                        <div class="kpi-label">Nachrichten</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['avg_duration_seconds']}s</div>
                        <div class="kpi-label">Ø Dauer</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['avg_messages_per_conversation']}</div>
                        <div class="kpi-label">Ø Msgs/Conv</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>✅ Erfolgsrate</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value success">{success['success_count']}</div>
                        <div class="kpi-label">Erfolg</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value neutral">{success['neutral_count']}</div>
                        <div class="kpi-label">Neutral</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value failure">{success['failure_count']}</div>
                        <div class="kpi-label">Misserfolg</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{success['success_rate']}%</div>
                        <div class="kpi-label">Erfolgsquote</div>
                    </div>
                </div>
                <div class="chart-container">
                    {fig_success.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Zeitlicher Verlauf</h2>
                <div class="chart-container">
                    {fig_timeline.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </div>
            
            <div class="section">
                <h2>🤖 Bot-Qualität</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">{helpless['helpless_count']}</div>
                        <div class="kpi-label">Hilflose Antworten</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{helpless['helpless_rate']}%</div>
                        <div class="kpi-label">Hilflosigkeits-Rate</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>💬 Top Phrasen</h2>
                <div class="chart-container">
                    {bigrams_html}
                </div>
            </div>
            
            <div class="section">
                <h2>❓ Häufigste Einstiegsfragen</h2>
                <table>
                    <tr><th>Frage</th><th>Anzahl</th></tr>
                    {"".join(f"<tr><td>{row['Frage'][:100]}...</td><td>{row['Anzahl']}</td></tr>" for _, row in first_qs.iterrows()) if not first_qs.empty else "<tr><td colspan='2'>Keine Daten</td></tr>"}
                </table>
            </div>
            
            <footer style="margin-top: 40px; padding: 20px; border-top: 1px solid #333; color: #666;">
                <p>Generiert mit Chatbase Analytics</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return html

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

# Export Button (now that filtered_analyzer is defined)
if st.sidebar.button("📄 HTML-Report generieren", help="Erstellt einen vollständigen Report als HTML-Datei zum Download."):
    with st.spinner("Generiere Report..."):
        html_report = generate_html_report(filtered_analyzer, start_date, end_date)
        st.sidebar.download_button(
            label="⬇️ HTML herunterladen",
            data=html_report,
            file_name=f"chatbase_report_{start_date}_{end_date}.html",
            mime="text/html"
        )

# Layout
st.title("🤖 Chatbot Conversation Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Überblick", "💬 Themen & Inhalte", "🧠 Qualität & Sentiment", "📂 Daten-Explorer"])

with tab1:
    st.header("Deskriptive Statistik", help="Allgemeine Kennzahlen zum Nachrichtenaufkommen.")
    stats = filtered_analyzer.get_basic_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Konversationen", stats['total_conversations'], help="Anzahl der geführten Gespräche im gewählten Zeitraum.")
    col2.metric("Nachrichten", stats['total_messages'], help="Gesamtzahl aller ausgetauschten Nachrichten (User + Bot).")
    col3.metric("Ø Dauer (sek)", stats['avg_duration_seconds'], help="Durchschnittliche Zeit zwischen erster und letzter Nachricht.")
    col4.metric("Ø Msgs/Conv", stats['avg_messages_per_conversation'], help="Wie viele Nachrichten werden durchschnittlich pro Gespräch ausgetauscht?")
    
    st.divider()
    
    # Timeline
    st.subheader("Entwicklung über Zeit", help="Zeigt, an welchen Tagen wie viele Gespräche stattgefunden haben.")
    
    # Toggle für Aggregation
    time_agg = st.radio("Zeitraum-Aggregation:", ["Täglich", "Wöchentlich"], horizontal=True,
                        help="Wähle, ob du die Daten pro Tag oder pro Woche sehen möchtest.")
    freq = 'D' if time_agg == "Täglich" else 'W'
    
    time_counts = filtered_analyzer.get_time_distribution(freq=freq)
    fig_timeline = px.line(time_counts, x='date', y='count', title=f'{time_agg}e Konversationen', markers=True)
    fig_timeline.update_layout(xaxis_title="Datum", yaxis_title="Anzahl")
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Heatmap (Weekday x Hour)
    st.subheader("Support-Auslastung (Heatmap)", help="Dunkle Felder zeigen Zeiten mit hoher Aktivität. Ideal für die Personalplanung.")
    heatmap_data = filtered_analyzer.get_heatmap_data()
    if not heatmap_data.empty:
        fig_heatmap = px.imshow(heatmap_data, 
                               labels=dict(x="Stunde", y="Wochentag", color="Anzahl"),
                               x=heatmap_data.columns,
                               y=heatmap_data.index,
                               title="Konversationen nach Zeit & Tag",
                               aspect="auto",
                               color_continuous_scale="YlOrRd")  # Gelb (wenig) -> Orange -> Rot (viel)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Zu wenig Daten für Heatmap.")

    # Source Distribution if available
    if 'source' in filtered_conv_df.columns and filtered_conv_df['source'].nunique() > 1:
        st.subheader("Quellen", help="Woher kommen die Konversationen? (z.B. Widget, API, etc.)")
        source_counts = filtered_conv_df['source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
        fig_source = px.pie(source_counts, values='count', names='source', title='Verteilung nach Quelle')
        st.plotly_chart(fig_source, use_container_width=True)

with tab2:
    st.header("Themen & Inhalte", help="Analyse, worüber die Nutzer sprechen und wie Gespräche verlaufen.")
    
    # Keyword Trends über Zeit
    st.subheader("Themen-Trend über Zeit", help="Zeigt, welche Keywords pro Woche/Monat am häufigsten genannt werden. Ideal um Trends und saisonale Themen zu erkennen.")
    
    trend_freq = st.radio("Trend-Aggregation:", ["Wöchentlich", "Monatlich"], horizontal=True,
                          help="Wähle, ob du die Keywords pro Woche oder pro Monat sehen möchtest.", key="trend_freq")
    trend_top_k = st.slider("Anzahl Top-Keywords:", 5, 15, 10, 
                            help="Wie viele der häufigsten Keywords sollen angezeigt werden?")
    
    freq_map = {'Wöchentlich': 'W', 'Monatlich': 'M'}
    trend_data = filtered_analyzer.get_keyword_trends(freq=freq_map[trend_freq], top_k=trend_top_k)
    
    if not trend_data.empty:
        fig_trend = px.line(trend_data, x='period', y='count', color='keyword',
                           title=f'Top {trend_top_k} Keywords im Zeitverlauf ({trend_freq})',
                           labels={'period': 'Zeitraum', 'count': 'Häufigkeit', 'keyword': 'Keyword'},
                           markers=True)
        fig_trend.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Nicht genügend Daten für Trend-Analyse.")
    
    st.divider()
    
    # Erste Fragen Analyse
    st.subheader("Häufigste Einstiegsfragen", help="Womit starten die Nutzer das Gespräch? Zeigt die Hauptanliegen.")
    first_questions = filtered_analyzer.get_first_questions(top_k=10)
    if not first_questions.empty:
        fig_first_q = px.bar(first_questions, x='Anzahl', y='Frage', orientation='h', 
                            title='Top 10 erste User-Nachrichten')
        fig_first_q.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_first_q, use_container_width=True)
    else:
        st.info("Keine Daten für Einstiegsfragen verfügbar.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Häufigste Phrasen (Bigrams)", help="Häufige Wortpaare (z.B. 'Konto eröffnen').")
        top_phrases = filtered_analyzer.get_top_phrases(n_gram_range=(2,2), top_k=10)
        if top_phrases:
            df_phrases = pd.DataFrame(top_phrases, columns=['Phrase', 'Count'])
            fig_bar = px.bar(df_phrases, x='Count', y='Phrase', orientation='h', title='Top 10 Bigrams')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Keine Daten für Phrasen verfügbar.")

    with col2:
        st.subheader("Häufigste Phrasen (Trigrams)", help="Häufige Wort-Trios (z.B. 'wie viel kostet').")
        top_trigrams = filtered_analyzer.get_top_phrases(n_gram_range=(3,3), top_k=10)
        if top_trigrams:
            df_tri = pd.DataFrame(top_trigrams, columns=['Phrase', 'Count'])
            fig_bar_tri = px.bar(df_tri, x='Count', y='Phrase', orientation='h', title='Top 10 Trigrams')
            fig_bar_tri.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar_tri, use_container_width=True)
        else:
            st.info("Keine Daten für Trigrams verfügbar.")
            
    st.divider()
    
    st.subheader("Topic Clustering", help="Gruppiert Gespräche automatisch in Themencluster.")
    
    # Auswahl der Methode
    clustering_method = st.radio(
        "Clustering-Methode:",
        ["K-Means (Lokal)", "🤖 AI-Analyse (OpenAI GPT)"],
        horizontal=True,
        help="K-Means ist kostenlos aber weniger genau. AI-Analyse nutzt OpenAI GPT für bessere semantische Erkennung (API-Kosten)."
    )
    
    if clustering_method == "K-Means (Lokal)":
        run_clustering = st.toggle("K-Means Clustering aktivieren", value=False,
                                   help="Schalte ein, um die Themen-Analyse zu starten.")
        
        if run_clustering:
            if 'cluster_data' not in st.session_state:
                st.session_state.cluster_data = None
            
            if st.session_state.cluster_data is None:
                with st.spinner("Analysiere Themen mit K-Means..."):
                    clustered_df, cluster_terms = filtered_analyzer.perform_topic_modeling(n_clusters=5)
                    st.session_state.cluster_data = (clustered_df, cluster_terms)
            
            clustered_df, cluster_terms = st.session_state.cluster_data
            
            cluster_stats = clustered_df.groupby('cluster').agg({
                'conversation_id': 'count',
                'duration_seconds': 'mean'
            }).reset_index()
            cluster_stats['Topic Terms'] = cluster_stats['cluster'].map(cluster_terms)
            
            st.dataframe(cluster_stats.style.format({'duration_seconds': '{:.1f}'}))
            
            fig_cluster = px.bar(cluster_stats, x='cluster', y='conversation_id', 
                                 hover_data=['Topic Terms'], 
                                 title='Konversationen pro Cluster',
                                 labels={'conversation_id': 'Anzahl', 'cluster': 'Cluster ID'})
            st.plotly_chart(fig_cluster, use_container_width=True)
    
    else:  # AI-Analyse
        st.info("🤖 **AI-Analyse** nutzt OpenAI GPT für präzise Themenerkennung. Stratifizierte Stichprobe aus bis zu 300 Konversationen für repräsentative Ergebnisse.")
        
        # Model selection
        ai_model = st.selectbox(
            "OpenAI Model:",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o: Beste Qualität | gpt-4o-mini: Günstig & schnell | gpt-4-turbo: Großes Context Window"
        )
        
        run_ai_clustering = st.toggle("AI Topic-Analyse starten", value=False,
                                      help="Startet die OpenAI-basierte Themenanalyse. Benötigt konfigurierten API-Key in Streamlit Secrets.")
        
        if run_ai_clustering:
            if 'ai_topics' not in st.session_state:
                st.session_state.ai_topics = None
            
            if st.session_state.ai_topics is None:
                with st.spinner(f"🤖 Analysiere Themen mit {ai_model}... (Dies kann 30-60 Sekunden dauern)"):
                    result = filtered_analyzer.perform_ai_topic_modeling(sample_size=500, model=ai_model)
                    st.session_state.ai_topics = result
            
            result = st.session_state.ai_topics
            
            if "error" in result:
                st.error(f"❌ Fehler: {result['error']}")
                st.info("💡 **Tipp:** Stelle sicher, dass der OpenAI API-Key in Streamlit Secrets konfiguriert ist:\n\n```toml\n[openai]\napi_key = \"sk-...\"\n```")
            else:
                # Success metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Analysiert", f"{result.get('conversations_analyzed', 0)} Konversationen")
                col2.metric("Abdeckung", f"{result.get('sample_coverage', 0)}%")
                col3.metric("Model", result.get('model_used', 'GPT'))
                
                st.caption(f"Sampling: {result.get('sampling_method', 'random')}")
                
                # Summary
                if "summary" in result:
                    st.markdown(f"### 📝 Executive Summary")
                    st.info(result['summary'])
                
                # Key Insights
                if "key_insights" in result and result["key_insights"]:
                    st.markdown("### 💡 Key Insights")
                    for insight in result["key_insights"]:
                        st.markdown(f"- {insight}")
                
                # Recommendations
                if "recommendations" in result and result["recommendations"]:
                    st.markdown("### 🎯 Empfehlungen")
                    for rec in result["recommendations"]:
                        st.markdown(f"- {rec}")
                
                st.markdown("### 🏷️ Erkannte Themen")
                
                # Display topics
                topics = result.get("topics", [])
                for i, topic in enumerate(topics, 1):
                    freq_color = {"hoch": "🔴", "mittel": "🟡", "niedrig": "🟢"}.get(topic.get("frequency", ""), "⚪")
                    sentiment_icon = {"positiv": "😊", "negativ": "😞", "neutral": "😐", "gemischt": "🔄"}.get(topic.get("sentiment_tendency", ""), "")
                    percentage = topic.get("estimated_percentage", "")
                    pct_str = f" ~{percentage}%" if percentage else ""
                    
                    with st.expander(f"{freq_color} **{topic.get('name', f'Thema {i}')}** ({topic.get('frequency', 'unbekannt')}{pct_str}) {sentiment_icon}"):
                        st.write(topic.get("description", "Keine Beschreibung"))
                        
                        if "example_keywords" in topic:
                            st.markdown(f"**Keywords:** `{'`, `'.join(topic['example_keywords'])}`")
                        
                        if "subtopics" in topic and topic["subtopics"]:
                            st.markdown(f"**Unterthemen:** {', '.join(topic['subtopics'])}")
                        
                        if "sentiment_tendency" in topic:
                            st.markdown(f"**Stimmungstendenz:** {topic['sentiment_tendency']}")
                
                # Reset button
                if st.button("🔄 Neue Analyse starten"):
                    st.session_state.ai_topics = None
                    st.rerun()
            
    st.divider()
    
    st.subheader("Word Cloud", help="Visuelle Darstellung der häufigsten Wörter. Je größer, desto öfter genannt.")
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
    st.subheader("Exit-Analyse (Wie enden Gespräche?)", help="Zeigt, ob der Nutzer (offen?) oder der Bot (geklärt?) das letzte Wort hatte.")
    
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
    st.header("Qualitäts-Analyse", help="Metriken zur Zufriedenheit und Komplexität der Anfragen.")
    
    # Erfolgsrate
    st.subheader("Erfolgsrate", help="Wurden die Nutzeranliegen gelöst? Basiert auf der letzten Nachricht jeder Konversation.")
    
    success_data = filtered_analyzer.calculate_success_rate()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Erfolg", success_data['success_count'], 
                help="User beendete mit positiven Keywords (danke, super, perfekt...)")
    col2.metric("⚪ Neutral", success_data['neutral_count'], 
                help="Normale Beendigung ohne eindeutige Signale.")
    col3.metric("❌ Misserfolg", success_data['failure_count'], 
                help="Bot konnte nicht helfen oder User war unzufrieden.")
    col4.metric("📈 Erfolgsquote", f"{success_data['success_rate']}%", 
                help="Anteil der erfolgreich abgeschlossenen Gespräche.")
    
    # Pie Chart für Erfolgsrate
    outcome_df = pd.DataFrame({
        'Outcome': ['Erfolg', 'Neutral', 'Misserfolg'],
        'Count': [success_data['success_count'], success_data['neutral_count'], success_data['failure_count']]
    })
    fig_success = px.pie(outcome_df, values='Count', names='Outcome', 
                         title='Gesprächs-Outcomes',
                         color='Outcome',
                         color_discrete_map={'Erfolg': '#2ECC71', 'Neutral': '#95A5A6', 'Misserfolg': '#E74C3C'})
    st.plotly_chart(fig_success, use_container_width=True)
    
    st.divider()
    
    st.subheader("Sentiment-Analyse", help="Analysiert die Stimmung der User-Nachrichten anhand positiver/negativer Keywords und Emojis.")
    
    # Session State für Sentiment
    if 'sentiment_data' not in st.session_state:
        st.session_state.sentiment_data = None
    
    # Toggle statt Button
    run_sentiment = st.toggle("Sentiment-Analyse aktivieren", value=False, 
                              help="Schalte ein, um die Stimmungsanalyse zu berechnen.")
    
    if run_sentiment:
        if st.session_state.sentiment_data is None:
            with st.spinner("Berechne Sentiment..."):
                st.session_state.sentiment_data = filtered_analyzer.analyze_sentiment()
        
        sentiment_df = st.session_state.sentiment_data
        
        # Erklärung der Skala
        st.markdown("""
        **So funktioniert die Sentiment-Analyse:**
        - Wir suchen nach **positiven Wörtern** (z.B. "danke", "super", "toll", 👍, 😊)
        - Wir suchen nach **negativen Wörtern** (z.B. "problem", "fehler", "geht nicht", 😞, 😡)
        - **Score:** -1 = nur negativ, 0 = neutral/gemischt, +1 = nur positiv
        """)
        
        st.divider()
        
        # Histogram mit Erklärung
        st.markdown("**📊 Grafik 1: Wie ist die Stimmung verteilt?**")
        st.caption("Jeder Balken zeigt, wie viele Konversationen einen bestimmten Sentiment-Wert haben. Balken links = unzufrieden, Balken rechts = zufrieden, Balken in der Mitte = neutral.")
        fig_hist = px.histogram(sentiment_df, x='sentiment', nbins=20, title='Sentiment-Verteilung')
        fig_hist.update_layout(xaxis_title="Sentiment-Score (-1 bis +1)", yaxis_title="Anzahl Konversationen")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.divider()
        
        # Scatter mit Erklärung
        st.markdown("**📈 Grafik 2: Hängt die Gesprächsdauer mit der Stimmung zusammen?**")
        st.caption("Jeder Punkt = eine Konversation. X-Achse = Dauer in Sekunden, Y-Achse = Sentiment. Muster erkennen: Sind lange Gespräche eher negativ?")
        fig_scatter = px.scatter(sentiment_df, x='duration_seconds', y='sentiment', 
                                 title='Gesprächsdauer vs. Stimmung', opacity=0.5)
        fig_scatter.update_layout(xaxis_title="Dauer (Sekunden)", yaxis_title="Sentiment-Score")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.subheader("Komplexitäts-Verteilung (Nachrichtenanzahl)", help="Zeigt, wie viele Chats kurz (Quick Fix) oder lang (komplex) sind.")
    
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

    st.divider()
    st.subheader("Bot-Hilflosigkeit", help="Wie oft signalisiert der Bot, dass er nicht helfen kann? (z.B. 'Ich weiß nicht', 'Support kontaktieren')")
    
    helpless_data = filtered_analyzer.get_bot_helplessness()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Hilflose Antworten", helpless_data['helpless_count'], help="Anzahl der Bot-Antworten mit Hilflosigkeits-Keywords.")
    col2.metric("Gesamt Bot-Nachrichten", helpless_data['total_bot_messages'], help="Gesamtzahl aller Bot-Antworten.")
    col3.metric("Hilflosigkeits-Rate", f"{helpless_data['helpless_rate']}%", 
                help="Prozentsatz der Bot-Antworten, die auf Wissenslücken hindeuten.")
    
    if not helpless_data['examples'].empty:
        st.markdown("**Top 5 'hilflose' Antworten:**")
        st.dataframe(helpless_data['examples'])
        
        # CSV Download für alle hilflosen Antworten
        if not helpless_data['all_helpless'].empty:
            csv = helpless_data['all_helpless'].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Alle hilflosen Antworten als CSV herunterladen",
                data=csv,
                file_name="hilflose_antworten.csv",
                mime="text/csv",
                help="Lädt alle als 'hilflos' erkannten Bot-Antworten herunter."
            )

    st.divider()
    st.subheader("Bot-Antwortlängen", help="Wie lang sind die Antworten des Bots im Durchschnitt? Zu kurz = nicht hilfreich, zu lang = verwirrend.")
    
    length_stats, length_df = filtered_analyzer.get_response_length_stats()
    
    col1, col2 = st.columns(2)
    col1.metric("Ø Zeichen pro Antwort", length_stats['avg_chars'], help="Durchschnittliche Zeichenanzahl pro Bot-Nachricht.")
    col2.metric("Ø Wörter pro Antwort", length_stats['avg_words'], help="Durchschnittliche Wortanzahl pro Bot-Nachricht.")
    
    fig_length = px.histogram(length_df, x='word_count', nbins=30, 
                             title='Verteilung der Antwortlänge (Wörter)',
                             labels={'word_count': 'Wörter pro Antwort'})
    st.plotly_chart(fig_length, use_container_width=True)

with tab4:
    st.header("Daten-Explorer", help="Durchsuche alle Nachrichten und Konversationen.")
    search_term = st.text_input("Suche in Nachrichten", help="Gib ein Suchwort ein, um alle passenden Nachrichten zu finden.")
    
    if search_term:
        results = filtered_msg_df[filtered_msg_df['content'].str.contains(search_term, case=False, na=False)]
        st.write(f"{len(results)} Treffer gefunden:")
        st.dataframe(results[['conversation_id', 'role', 'content']])
    else:
        st.subheader("Konversationen")
        st.dataframe(filtered_conv_df)
        st.subheader("Nachrichten")
        st.dataframe(filtered_msg_df.head(100))
