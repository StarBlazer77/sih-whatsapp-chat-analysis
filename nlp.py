# nlp.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[~df['message'].str.contains('<Media omitted>')]
    df['sentiment_score'] = df['message'].apply(lambda x: sia.polarity_scores(x)['compound'])
    # aggregate
    avg = df['sentiment_score'].mean()
    pos = (df['sentiment_score'] > 0.05).mean()
    neg = (df['sentiment_score'] < -0.05).mean()
    neutral = 1 - pos - neg
    return {'avg': avg, 'pos_pct': pos, 'neg_pct': neg, 'neutral_pct': neutral}, df[['date','user','message','sentiment_score']].head(50)


