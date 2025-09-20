# graph.py
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os


# ---------- RESPONSE TIMES ----------
def response_times(df):
    """
    Calculate response times between consecutive messages.
    Returns a DataFrame of interactions and per-user median response time in minutes.
    """
    df = df.sort_values('date').reset_index(drop=True)
    rows = []

    for i in range(1, len(df)):
        prev_user, curr_user = df.loc[i - 1, 'user'], df.loc[i, 'user']
        delta = df.loc[i, 'date'] - df.loc[i - 1, 'date']
        if prev_user != curr_user:
            rows.append({
                'from': prev_user,
                'to': curr_user,
                'response_time_s': delta.total_seconds()
            })

    rt = pd.DataFrame(rows)
    if rt.empty:
        return rt, pd.Series(dtype=float)

    per_user = rt.groupby('to')['response_time_s'].median().apply(lambda x: x / 60)
    return rt, per_user.sort_values()


# ---------- INTERACTION GRAPH ----------
def build_interaction_graph(rt_df):
    """
    Build a directed interaction graph from response times DataFrame.
    """
    if rt_df.empty:
        return nx.DiGraph()

    pairs = rt_df.groupby(['from', 'to']).size().reset_index(name='weight')
    g = nx.DiGraph()
    for _, row in pairs.iterrows():
        g.add_edge(row['from'], row['to'], weight=row['weight'])
    return g


# ---------- PYVIS PLOT ----------
def plot_pyvis(g, height='600px'):
    """
    Render a pyvis interactive graph in Streamlit.
    """
    if len(g.nodes) == 0:
        return

    net = Network(height=height, width='100%', notebook=False, directed=True)
    net.from_nx(g)
    net.force_atlas_2based()
    net.save_graph('graph.html')

    # Properly read the HTML and render
    with open('graph.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=int(height.replace('px', '')), scrolling=True)

    # Remove temporary HTML file
    if os.path.exists('graph.html'):
        os.remove('graph.html')
