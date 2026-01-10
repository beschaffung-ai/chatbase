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

def generate_html_report(analyzer_obj, start_d, end_d, conv_df):
    """Generates a comprehensive HTML report with ALL analyses and professional design."""
    import base64
    from io import BytesIO
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    # ===== COLLECT ALL DATA =====
    stats = analyzer_obj.get_basic_stats()
    success = analyzer_obj.calculate_success_rate()
    helpless = analyzer_obj.get_bot_helplessness()
    time_dist_daily = analyzer_obj.get_time_distribution(freq='D')
    time_dist_weekly = analyzer_obj.get_time_distribution(freq='W')
    first_qs = analyzer_obj.get_first_questions(top_k=10)
    top_bigrams = analyzer_obj.get_top_phrases(n_gram_range=(2,2), top_k=10)
    top_trigrams = analyzer_obj.get_top_phrases(n_gram_range=(3,3), top_k=10)
    heatmap_data = analyzer_obj.get_heatmap_data()
    exit_df = analyzer_obj.get_exit_analysis()
    length_stats, length_df = analyzer_obj.get_response_length_stats()
    sentiment_df = analyzer_obj.analyze_sentiment()
    keyword_trends = analyzer_obj.get_keyword_trends(freq='W', top_k=8)
    
    # ===== GENERATE CHARTS =====
    # Timeline
    fig_timeline = px.line(time_dist_weekly, x='date', y='count', title='Wöchentliche Konversationen', markers=True)
    fig_timeline.update_layout(
        xaxis_title="Datum", yaxis_title="Anzahl",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaeaea')
    )
    
    # Success Pie
    outcome_df = pd.DataFrame({
        'Outcome': ['Erfolg', 'Neutral', 'Misserfolg'],
        'Count': [success['success_count'], success['neutral_count'], success['failure_count']]
    })
    fig_success = px.pie(outcome_df, values='Count', names='Outcome', title='Gesprächs-Outcomes',
                         color='Outcome', color_discrete_map={'Erfolg': '#2ECC71', 'Neutral': '#95A5A6', 'Misserfolg': '#E74C3C'})
    fig_success.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
    
    # Heatmap
    if not heatmap_data.empty:
        fig_heatmap = px.imshow(heatmap_data, 
                               labels=dict(x="Stunde", y="Wochentag", color="Anzahl"),
                               x=heatmap_data.columns, y=heatmap_data.index,
                               title="Support-Auslastung nach Wochentag & Uhrzeit",
                               aspect="auto", color_continuous_scale="YlOrRd")
        fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
        heatmap_html = fig_heatmap.to_html(full_html=False, include_plotlyjs=False)
    else:
        heatmap_html = "<p style='color: #888;'>Nicht genügend Daten für Heatmap.</p>"
    
    # Bigrams
    if top_bigrams:
        df_bigrams = pd.DataFrame(top_bigrams, columns=['Phrase', 'Count'])
        fig_bigrams = px.bar(df_bigrams, x='Count', y='Phrase', orientation='h', title='Top 10 Häufige Wortpaare (Bigrams)')
        fig_bigrams.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', 
                                  plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
        bigrams_html = fig_bigrams.to_html(full_html=False, include_plotlyjs=False)
    else:
        bigrams_html = "<p style='color: #888;'>Keine Daten für Bigrams.</p>"
    
    # Trigrams
    if top_trigrams:
        df_trigrams = pd.DataFrame(top_trigrams, columns=['Phrase', 'Count'])
        fig_trigrams = px.bar(df_trigrams, x='Count', y='Phrase', orientation='h', title='Top 10 Wort-Trios (Trigrams)')
        fig_trigrams.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
        trigrams_html = fig_trigrams.to_html(full_html=False, include_plotlyjs=False)
    else:
        trigrams_html = "<p style='color: #888;'>Keine Daten für Trigrams.</p>"
    
    # First Questions
    if not first_qs.empty:
        fig_first_q = px.bar(first_qs.head(10), x='Anzahl', y='Frage', orientation='h', title='Top 10 Einstiegsfragen')
        fig_first_q.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
        first_q_html = fig_first_q.to_html(full_html=False, include_plotlyjs=False)
    else:
        first_q_html = "<p style='color: #888;'>Keine Daten.</p>"
    
    # Exit Analysis
    last_role_counts = exit_df['role'].value_counts().reset_index()
    last_role_counts.columns = ['Role', 'Count']
    fig_exit = px.pie(last_role_counts, values='Count', names='Role', title='Wer sendet die letzte Nachricht?',
                      color='Role', color_discrete_map={'user': '#FF6B6B', 'assistant': '#4ECDC4'})
    fig_exit.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
    
    # Message Count Distribution
    fig_msg_dist = px.histogram(conv_df, x='message_count', nbins=30, 
                                title='Konversationslänge (Nachrichtenanzahl)',
                                labels={'message_count': 'Anzahl Nachrichten'})
    fig_msg_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
    
    # Sentiment Histogram
    fig_sentiment = px.histogram(sentiment_df, x='sentiment', nbins=20, title='Sentiment-Verteilung',
                                 labels={'sentiment': 'Sentiment-Score (-1 bis +1)'})
    fig_sentiment.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
    
    # Bot Response Length
    fig_length = px.histogram(length_df, x='word_count', nbins=30, title='Bot-Antwortlängen (Wörter)',
                              labels={'word_count': 'Wörter pro Antwort'})
    fig_length.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
    
    # Keyword Trends
    if not keyword_trends.empty:
        fig_trends = px.line(keyword_trends, x='period', y='count', color='keyword', markers=True,
                            title='Keyword-Trends über Zeit', labels={'period': 'Woche', 'count': 'Häufigkeit'})
        fig_trends.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'),
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02))
        trends_html = fig_trends.to_html(full_html=False, include_plotlyjs=False)
    else:
        trends_html = "<p style='color: #888;'>Nicht genügend Daten.</p>"
    
    # Word Cloud as Base64 Image
    wordcloud_text = analyzer_obj.get_wordcloud_text()
    if wordcloud_text:
        wc = WordCloud(width=800, height=400, background_color='#0d1b2a', colormap='viridis',
                       stopwords=analyzer_obj.custom_stopwords).generate(wordcloud_text)
        buf = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(buf, format='png', facecolor='#0d1b2a', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        wc_base64 = base64.b64encode(buf.read()).decode()
        wordcloud_html = f'<img src="data:image/png;base64,{wc_base64}" style="width: 100%; border-radius: 12px;" alt="Word Cloud">'
    else:
        wordcloud_html = "<p style='color: #888;'>Nicht genügend Text für Wordcloud.</p>"
    
    # Source Distribution
    source_html = ""
    if 'source' in conv_df.columns and conv_df['source'].nunique() > 1:
        source_counts = conv_df['source'].value_counts().reset_index()
        source_counts.columns = ['source', 'count']
        fig_source = px.pie(source_counts, values='count', names='source', title='Quellenverteilung')
        fig_source.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
        source_html = f'<div class="chart-box">{fig_source.to_html(full_html=False, include_plotlyjs=False)}</div>'
    
    # Komplexitäts-Klassifizierung
    def classify_length(n):
        if n <= 3: return "Kurz (1-3)"
        if n <= 10: return "Mittel (4-10)"
        return "Lang (>10)"
    conv_df_copy = conv_df.copy()
    conv_df_copy['length_class'] = conv_df_copy['message_count'].apply(classify_length)
    class_counts = conv_df_copy['length_class'].value_counts().reset_index()
    class_counts.columns = ['Klasse', 'Anzahl']
    fig_complexity = px.pie(class_counts, values='Anzahl', names='Klasse', title='Gesprächskomplexität',
                            color='Klasse', color_discrete_map={'Kurz (1-3)': '#2ECC71', 'Mittel (4-10)': '#F39C12', 'Lang (>10)': '#E74C3C'})
    fig_complexity.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#eaeaea'))
    
    # ===== BUILD HTML =====
    from datetime import datetime
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbase Analytics Report</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-primary: #0d1b2a;
                --bg-secondary: #1b263b;
                --bg-card: #1e3a5f;
                --accent: #00d4ff;
                --accent-secondary: #7b2cbf;
                --success: #2ECC71;
                --warning: #F39C12;
                --danger: #E74C3C;
                --text-primary: #eaeaea;
                --text-secondary: #a0aec0;
                --border: #2d3748;
            }}
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, var(--bg-primary) 0%, #1a1a2e 50%, var(--bg-secondary) 100%);
                color: var(--text-primary);
                min-height: 100vh;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                padding: 40px 20px;
            }}
            
            /* Header */
            .header {{
                text-align: center;
                padding: 60px 20px;
                background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(123, 44, 191, 0.1) 100%);
                border-radius: 20px;
                margin-bottom: 40px;
                border: 1px solid var(--border);
            }}
            .header h1 {{
                font-size: 3em;
                font-weight: 700;
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-secondary) 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 10px;
            }}
            .header .subtitle {{
                color: var(--text-secondary);
                font-size: 1.2em;
            }}
            .header .date-range {{
                margin-top: 20px;
                padding: 10px 20px;
                background: var(--bg-secondary);
                border-radius: 30px;
                display: inline-block;
                font-weight: 500;
            }}
            
            /* Sections */
            .section {{
                margin-bottom: 50px;
            }}
            .section-title {{
                font-size: 1.8em;
                font-weight: 600;
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 2px solid var(--border);
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .section-title .emoji {{
                font-size: 1.2em;
            }}
            
            /* KPI Grid */
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .kpi-card {{
                background: linear-gradient(145deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
                border-radius: 16px;
                padding: 25px;
                text-align: center;
                border: 1px solid var(--border);
                transition: transform 0.3s, box-shadow 0.3s;
            }}
            .kpi-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0, 212, 255, 0.1);
            }}
            .kpi-value {{
                font-size: 2.5em;
                font-weight: 700;
                color: var(--accent);
                margin-bottom: 5px;
            }}
            .kpi-value.success {{ color: var(--success); }}
            .kpi-value.warning {{ color: var(--warning); }}
            .kpi-value.danger {{ color: var(--danger); }}
            .kpi-label {{
                color: var(--text-secondary);
                font-size: 0.95em;
                font-weight: 500;
            }}
            
            /* Chart Box */
            .chart-box {{
                background: var(--bg-secondary);
                border-radius: 16px;
                padding: 25px;
                margin-bottom: 25px;
                border: 1px solid var(--border);
            }}
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 25px;
            }}
            @media (max-width: 600px) {{
                .chart-grid {{ grid-template-columns: 1fr; }}
            }}
            
            /* Table */
            .data-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                background: var(--bg-secondary);
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--border);
            }}
            .data-table th {{
                background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
                padding: 15px;
                text-align: left;
                font-weight: 600;
                color: var(--accent);
            }}
            .data-table td {{
                padding: 12px 15px;
                border-top: 1px solid var(--border);
            }}
            .data-table tr:hover td {{
                background: rgba(0, 212, 255, 0.05);
            }}
            
            /* Footer */
            .footer {{
                text-align: center;
                padding: 40px;
                margin-top: 60px;
                border-top: 1px solid var(--border);
                color: var(--text-secondary);
            }}
            .footer .logo {{
                font-size: 1.5em;
                font-weight: 600;
                color: var(--accent);
                margin-bottom: 10px;
            }}
            
            /* Print Styles */
            @media print {{
                body {{ background: white; color: black; }}
                .kpi-card, .chart-box {{ border: 1px solid #ccc; }}
                .kpi-value {{ color: #0066cc; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <header class="header">
                <h1>📊 Chatbase Analytics Report</h1>
                <p class="subtitle">Umfassende Analyse deiner Chatbot-Konversationen</p>
                <div class="date-range">
                    📅 Zeitraum: {start_d} bis {end_d}
                </div>
            </header>
            
            <!-- Section 1: Overview -->
            <section class="section">
                <h2 class="section-title"><span class="emoji">📈</span> Übersicht</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['total_conversations']:,}</div>
                        <div class="kpi-label">Konversationen</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['total_messages']:,}</div>
                        <div class="kpi-label">Nachrichten</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['avg_duration_seconds']}s</div>
                        <div class="kpi-label">Ø Dauer</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{stats['avg_messages_per_conversation']}</div>
                        <div class="kpi-label">Ø Nachrichten/Gespräch</div>
                    </div>
                </div>
            </section>
            
            <!-- Section 2: Success Rate -->
            <section class="section">
                <h2 class="section-title"><span class="emoji">✅</span> Erfolgsrate</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value success">{success['success_count']:,}</div>
                        <div class="kpi-label">Erfolg</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{success['neutral_count']:,}</div>
                        <div class="kpi-label">Neutral</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value danger">{success['failure_count']:,}</div>
                        <div class="kpi-label">Misserfolg</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value" style="color: {'var(--success)' if success['success_rate'] >= 60 else 'var(--warning)' if success['success_rate'] >= 40 else 'var(--danger)'};">{success['success_rate']}%</div>
                        <div class="kpi-label">Erfolgsquote</div>
                    </div>
                </div>
                <div class="chart-box">
                    {fig_success.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </section>
            
            <!-- Section 3: Bot Quality -->
            <section class="section">
                <h2 class="section-title"><span class="emoji">🤖</span> Bot-Qualität</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-value warning">{helpless['helpless_count']:,}</div>
                        <div class="kpi-label">Hilflose Antworten</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{helpless['total_bot_messages']:,}</div>
                        <div class="kpi-label">Bot-Nachrichten gesamt</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value" style="color: {'var(--success)' if helpless['helpless_rate'] <= 5 else 'var(--warning)' if helpless['helpless_rate'] <= 15 else 'var(--danger)'};">{helpless['helpless_rate']}%</div>
                        <div class="kpi-label">Hilflosigkeits-Rate</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{length_stats['avg_words']}</div>
                        <div class="kpi-label">Ø Wörter/Antwort</div>
                    </div>
                </div>
                <div class="chart-grid">
                    <div class="chart-box">
                        {fig_length.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                    <div class="chart-box">
                        {fig_exit.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>
            </section>
            
            <!-- Section 4: Time Analysis -->
            <section class="section">
                <h2 class="section-title"><span class="emoji">📅</span> Zeitliche Analyse</h2>
                <div class="chart-box">
                    {fig_timeline.to_html(full_html=False, include_plotlyjs=False)}
                </div>
                <div class="chart-box">
                    {heatmap_html}
                </div>
                {source_html}
            </section>
            
            <!-- Section 5: Content Analysis -->
            <section class="section">
                <h2 class="section-title"><span class="emoji">💬</span> Themen & Inhalte</h2>
                <div class="chart-box">
                    {first_q_html}
                </div>
                <div class="chart-grid">
                    <div class="chart-box">
                        {bigrams_html}
                    </div>
                    <div class="chart-box">
                        {trigrams_html}
                    </div>
                </div>
                <div class="chart-box">
                    <h3 style="margin-bottom: 20px; color: var(--accent);">☁️ Word Cloud</h3>
                    {wordcloud_html}
                </div>
            </section>
            
            <!-- Section 6: Trends -->
            <section class="section">
                <h2 class="section-title"><span class="emoji">📊</span> Keyword-Trends über Zeit</h2>
                <div class="chart-box">
                    {trends_html}
                </div>
            </section>
            
            <!-- Section 7: Complexity & Sentiment -->
            <section class="section">
                <h2 class="section-title"><span class="emoji">🧠</span> Komplexität & Sentiment</h2>
                <div class="chart-grid">
                    <div class="chart-box">
                        {fig_msg_dist.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                    <div class="chart-box">
                        {fig_complexity.to_html(full_html=False, include_plotlyjs=False)}
                    </div>
                </div>
                <div class="chart-box">
                    {fig_sentiment.to_html(full_html=False, include_plotlyjs=False)}
                </div>
            </section>
            
            <!-- Footer -->
            <footer class="footer">
                <div class="logo">🤖 Chatbase Analytics</div>
                <p>Report generiert am {generated_at}</p>
                <p style="margin-top: 10px; font-size: 0.9em;">Powered by Streamlit, Plotly & Python</p>
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
if st.sidebar.button("📄 HTML-Report generieren", help="Erstellt einen vollständigen Report mit ALLEN Analysen als interaktive HTML-Datei."):
    with st.spinner("Generiere umfassenden Report... (kann einige Sekunden dauern)"):
        html_report = generate_html_report(filtered_analyzer, start_date, end_date, filtered_conv_df)
        st.sidebar.download_button(
            label="⬇️ HTML herunterladen",
            data=html_report,
            file_name=f"chatbase_report_{start_date}_{end_date}.html",
            mime="text/html"
        )
        st.sidebar.success("✅ Report generiert!")

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
        ["K-Means (Lokal, kostenlos)", "🧠 Embedding (Alle Daten, empfohlen)"],
        horizontal=True,
        help="K-Means: kostenlos, schnell, weniger genau | Embedding: ALLE Daten mit OpenAI Embeddings + GPT (~$0.07)"
    )
    
    if clustering_method == "K-Means (Lokal, kostenlos)":
        st.info("⚡ **K-Means** ist kostenlos und schnell, aber weniger genau als Embedding-Clustering.")
        
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
    
    else:  # Embedding Clustering (Alle Daten)
        st.success("🧠 **Embedding Clustering** analysiert **ALLE** Konversationen mit OpenAI Embeddings + GPT-Beschreibungen.")
        
        st.markdown("""
        **So funktioniert es:**
        1. 📊 Alle Konversationen werden in semantische Vektoren umgewandelt (Embeddings)
        2. 🔍 UMAP reduziert die Dimensionen für besseres Clustering
        3. 🎯 HDBSCAN findet automatisch die Themengruppen
        4. 🤖 GPT-4o-mini benennt und beschreibt jeden Cluster
        
        **Kosten:** ~$0.05-0.10 für 5.000 Konversationen
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            use_hdbscan = st.checkbox("HDBSCAN (automatische Cluster-Anzahl)", value=True,
                                      help="HDBSCAN findet automatisch die optimale Anzahl an Clustern.")
        with col2:
            if not use_hdbscan:
                n_clusters_embed = st.slider("Anzahl Cluster (K-Means):", 3, 20, 8)
            else:
                n_clusters_embed = 8  # Wird ignoriert bei HDBSCAN
        
        run_embedding_clustering = st.toggle("🚀 Embedding-Analyse starten", value=False,
                                             help="Startet die vollständige Embedding-Analyse aller Konversationen.")
        
        if run_embedding_clustering:
            if 'embedding_result' not in st.session_state:
                st.session_state.embedding_result = None
            
            if st.session_state.embedding_result is None:
                progress_bar = st.progress(0, text="Starte Analyse...")
                status_text = st.empty()
                
                def update_progress(value, text):
                    progress_bar.progress(value, text=text)
                    status_text.text(text)
                
                result = filtered_analyzer.perform_embedding_clustering(
                    n_clusters=n_clusters_embed,
                    use_hdbscan=use_hdbscan,
                    progress_callback=update_progress
                )
                st.session_state.embedding_result = result
                progress_bar.empty()
                status_text.empty()
            
            result = st.session_state.embedding_result
            
            if "error" in result:
                st.error(f"❌ Fehler: {result['error']}")
            else:
                # Success metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("✅ Analysiert", f"{result.get('total_analyzed', 0):,}")
                col2.metric("📊 Cluster", result.get('n_clusters', 0))
                col3.metric("🔧 Methode", result.get('method', 'HDBSCAN'))
                col4.metric("🎯 Abdeckung", "100%")
                
                if result.get('noise_count', 0) > 0:
                    st.caption(f"ℹ️ {result['noise_count']} Konversationen konnten keinem Cluster zugeordnet werden (Noise)")
                
                # ===== SCATTER PLOT =====
                st.markdown("### 🗺️ Themen-Landkarte (UMAP 2D)")
                st.caption("Jeder Punkt ist eine Konversation. Ähnliche Themen liegen nah beieinander.")
                
                viz_data = result.get('visualization_data', {})
                if viz_data:
                    scatter_df = pd.DataFrame({
                        'x': viz_data['x'],
                        'y': viz_data['y'],
                        'cluster': viz_data['labels'],
                        'text': viz_data['texts']
                    })
                    # Map cluster IDs to names
                    cluster_names = {cid: desc.get('name', f'Cluster {cid}') 
                                    for cid, desc in result.get('cluster_descriptions', {}).items()}
                    cluster_names[-1] = 'Nicht zugeordnet'
                    scatter_df['cluster_name'] = scatter_df['cluster'].map(lambda x: cluster_names.get(x, f'Cluster {x}'))
                    
                    fig_scatter = px.scatter(
                        scatter_df, x='x', y='y', color='cluster_name',
                        hover_data={'text': True, 'x': False, 'y': False, 'cluster': False, 'cluster_name': False},
                        title='Konversationen im semantischen Raum',
                        labels={'cluster_name': 'Thema'},
                        opacity=0.7
                    )
                    fig_scatter.update_traces(marker=dict(size=6))
                    fig_scatter.update_layout(
                        xaxis_title="", yaxis_title="",
                        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02)
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # ===== CLUSTER OVERVIEW =====
                st.markdown("### 🏷️ Erkannte Themen")
                
                cluster_info = result.get('cluster_info', {})
                cluster_descriptions = result.get('cluster_descriptions', {})
                
                # Sort clusters by size
                sorted_clusters = sorted(
                    [(cid, info) for cid, info in cluster_info.items() if cid >= 0],
                    key=lambda x: x[1]['size'],
                    reverse=True
                )
                
                for cluster_id, info in sorted_clusters:
                    desc = cluster_descriptions.get(cluster_id, {})
                    name = desc.get('name', f'Thema {cluster_id + 1}')
                    description = desc.get('description', 'Keine Beschreibung verfügbar')
                    sentiment = desc.get('sentiment', 'neutral')
                    sentiment_icon = {"positiv": "😊", "negativ": "😞", "neutral": "😐"}.get(sentiment, "😐")
                    
                    with st.expander(f"**{name}** — {info['size']:,} Konversationen ({info['percentage']}%) {sentiment_icon}"):
                        st.write(description)
                        
                        if info.get('keywords'):
                            st.markdown(f"**Keywords:** `{'`, `'.join(info['keywords'][:8])}`")
                        
                        if info.get('examples'):
                            st.markdown("**Beispiele:**")
                            for ex in info['examples'][:3]:
                                st.markdown(f"- _{ex}_")
                
                # ===== BAR CHART =====
                st.markdown("### 📊 Cluster-Größenverteilung")
                bar_data = pd.DataFrame([
                    {'Thema': cluster_descriptions.get(cid, {}).get('name', f'Cluster {cid}'), 
                     'Anzahl': info['size'],
                     'Prozent': info['percentage']}
                    for cid, info in sorted_clusters
                ])
                if not bar_data.empty:
                    fig_bar = px.bar(bar_data, x='Anzahl', y='Thema', orientation='h',
                                     text='Prozent', title='Konversationen pro Thema')
                    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Reset button
                if st.button("🔄 Neue Embedding-Analyse starten"):
                    st.session_state.embedding_result = None
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
