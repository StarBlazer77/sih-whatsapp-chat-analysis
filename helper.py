from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import re
import emoji
import tldextract
from textblob import TextBlob
from detoxify import Detoxify
from better_profanity import profanity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import hashlib

extract = URLExtract()
profanity.load_censor_words()

SHORTENER_DOMAINS = {'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly'}

# ---------- BASIC STATS ----------
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = sum(len(message.split()) for message in df['message'])
    num_media_messages = df['message'].str.contains('<Media omitted>|<Media omitted>\n', regex=True).sum()
    links = sum(len(extract.find_urls(message)) for message in df['message'])
    return num_messages, words, num_media_messages, links

# ---------- WORD ANALYSIS ----------
def create_wordcloud(selected_user, df):
    stop_words = []
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = f.read().splitlines()
    except FileNotFoundError:
        pass

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != '<Media omitted>\n']

    temp['cleaned_message'] = temp['message'].apply(
        lambda msg: " ".join([w for w in msg.lower().split() if w not in stop_words])
    )

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(temp['cleaned_message'].str.cat(sep=" "))


def most_common_words(selected_user, df):
    stop_words = []
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8') as f:
            stop_words = f.read().splitlines()
    except FileNotFoundError:
        pass

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification'].copy()
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            word = re.sub(r'\W+', '', word)
            if word and word not in stop_words:
                words.append(word)

    return pd.DataFrame(Counter(words).most_common(20))

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [c for message in df['message'] for c in message if emoji.is_emoji(c)]
    return pd.DataFrame(Counter(emojis).most_common(20))

# ---------- TIMELINES ----------
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline['month'] + "_" + timeline['year'].astype(str)
    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()

# ---------- ACTIVITY ----------
def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

# ---------- NLP ----------
def sentiment_analysis(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    sentiments = df['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return sentiments.mean(), sentiments

def toxicity_analysis(selected_user, df):
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    return Detoxify('original').predict(" ".join(df['message'].astype(str)))

def detect_toxic(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df.copy()
    df['is_toxic'] = df['message'].apply(profanity.contains_profanity)
    return df['is_toxic'].sum(), df[df['is_toxic']].head(20)

# ---------- TOPIC MODELING ----------
def topic_modeling(selected_user, df, n_topics=5, n_top_words=8):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    texts = df['message'].astype(str).tolist()
    if len(texts) < 5:
        return [("Not enough data", [])]

    vec = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)

    topics = []
    for i, topic in enumerate(lda.components_):
        words = [vec.get_feature_names_out()[j] for j in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append((f"Topic {i+1}", words))
    return topics

# ---------- URL CHECK ----------
def check_urls(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    urls = [u for msg in df['message'] for u in extract.find_urls(msg)]
    url_info = []
    for u in urls:
        te = tldextract.extract(u)
        domain = f"{te.domain}.{te.suffix}" if te.suffix else te.domain
        is_short = domain in SHORTENER_DOMAINS
        is_ip = bool(re.match(r'^\d+\.\d+\.\d+\.\d+$', te.domain))
        suspicious = is_short or is_ip or (len(u) > 200) or ('@' in u)
        url_info.append({'url': u, 'domain': domain, 'suspicious': suspicious})

    return pd.DataFrame(url_info).drop_duplicates()

# ---------- RESPONSE TIME ----------
def response_times(df):
    df = df.sort_values('date').reset_index(drop=True)
    rows = []
    for i in range(1, len(df)):
        prev_user, curr_user = df.loc[i-1, 'user'], df.loc[i, 'user']
        delta = df.loc[i, 'date'] - df.loc[i-1, 'date']
        if prev_user != curr_user:
            rows.append({'from': prev_user, 'to': curr_user, 'response_time_s': delta.total_seconds()})

    rt = pd.DataFrame(rows)
    if rt.empty:
        return rt, pd.Series(dtype=float)

    per_user = rt.groupby('to')['response_time_s'].median().apply(lambda x: x/60)
    return rt, per_user.sort_values()

# ---------- USER INTERACTION GRAPH ----------
def build_interaction_graph(rt_df):
    if rt_df.empty:
        return nx.DiGraph()
    pairs = rt_df.groupby(['from', 'to']).size().reset_index(name='weight')
    g = nx.DiGraph()
    for _, row in pairs.iterrows():
        g.add_edge(row['from'], row['to'], weight=row['weight'])
    return g

def plot_pyvis(g, height='600px'):
    if len(g.nodes) == 0:
        return
    net = Network(height=height, width='100%', notebook=False, directed=True)
    net.from_nx(g)
    net.force_atlas_2based()
    net.save_graph('graph.html')

    # Properly open and close file to remove warnings
    with open('graph.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=int(height.replace('px', '')), scrolling=True)

    if os.path.exists('graph.html'):
        os.remove('graph.html')

# ---------- ANONYMIZATION ----------
def anonymize_df(df, keep_map=False):
    mapping = {u: f"User_{hashlib.sha256(u.encode()).hexdigest()[:8]}" for u in df['user'].unique()}
    df2 = df.copy()
    df2['user'] = df2['user'].map(mapping)
    return (df2, mapping) if keep_map else df2
