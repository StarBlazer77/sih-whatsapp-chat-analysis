# app.py
import streamlit as st
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

import helper  # Your fixed helper module
import security  # anonymization + encryption

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📊 WhatsApp Chat Analyzer")

# ---------- Sidebar ----------
st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload WhatsApp Chat File")

anon = st.sidebar.checkbox("Anonymize Users (Recommended)", value=True)
run_nlp = st.sidebar.checkbox("Run Advanced NLP Analysis (Sentiment, Toxicity, Topics)", value=True)
save_encrypted = st.sidebar.checkbox("Save Encrypted Report", value=False)

# ---------- Main App ----------
if uploaded_file:
    # Read file
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    # Preprocess using your preprocessor
    import preprocessor
    df = preprocessor.preprocess(data)

    # Optional anonymization
    if anon:
        df = security.anonymize_df(df)

    # User selection
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Select User for Analysis", user_list)

    # ---------- Run Full Analysis ----------
    if st.sidebar.button("Run Full Analysis"):
        with st.spinner("Running analytics..."):
            # ---------- Basic Stats ----------
            num_messages, words, media, links = helper.fetch_stats(selected_user, df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Messages", num_messages)
            col2.metric("Words", words)
            col3.metric("Media Shared", media)
            col4.metric("Links Shared", links)

            # ---------- Monthly Timeline ----------
            st.subheader("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ---------- Daily Timeline ----------
            st.subheader("Daily Timeline")
            daily = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily['only_date'], daily['message'], color='blue')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ---------- Activity ----------
            st.subheader("Activity Heatmaps")
            col1, col2 = st.columns(2)
            with col1:
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='orange')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            with col2:
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='purple')
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # ---------- WordCloud ----------
            st.subheader("Word Cloud")
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

            # ---------- Most Common Words ----------
            st.subheader("Most Common Words")
            most_common_df = helper.most_common_words(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(most_common_df[0], most_common_df[1])
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # ---------- Emoji Analysis ----------
            st.subheader("Emoji Analysis")
            emoji_df = helper.emoji_helper(selected_user, df)
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1], labels=emoji_df[0], autopct="%1.1f%%")
                st.pyplot(fig)

            # ---------- NLP Analysis ----------
            if run_nlp:
                st.subheader("Sentiment Analysis")
                avg_sent, _ = helper.sentiment_analysis(selected_user, df)
                st.write("Group Mood:",
                         "😊 Positive" if avg_sent > 0 else "😐 Neutral" if avg_sent == 0 else "😠 Negative")

                st.subheader("Toxic Messages")
                toxic_count, toxic_examples = helper.detect_toxic(selected_user, df)
                st.write(f"Total Toxic Messages: {toxic_count}")
                st.dataframe(toxic_examples[['user', 'message']])

                st.subheader("Topic Modeling")
                topics = helper.topic_modeling(selected_user, df)
                for topic, words in topics:
                    st.write(f"**{topic}:** {', '.join(words)}")

            # ---------- User Interaction Network ----------
            st.subheader("User Interaction Network")
            rt_df, _ = helper.response_times(df)
            G = helper.build_interaction_graph(rt_df)
            helper.plot_pyvis(G)

            # ---------- Save Encrypted Report ----------
            if save_encrypted:
                report_bytes = df.to_csv(index=False).encode()
                encrypted_bytes = security.encrypt_bytes(report_bytes)
                with open("encrypted_report.bin", "wb") as f:
                    f.write(encrypted_bytes)
                st.success("Encrypted report saved as encrypted_report.bin")

st.sidebar.info("Upload your WhatsApp chat export and run full analysis!")
