import pandas as pd
import numpy as np
import yfinance as yf
import praw
from newsapi import NewsApiClient
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import re
import warnings


warnings.filterwarnings('ignore')


load_dotenv()

class StockPredictor:
    def __init__(self, stock_symbol):
        """Initialize the StockPredictor with API clients"""
        self.stock_symbol = stock_symbol
        try:
            self.news_api = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        except:
            print("Warning: NewsAPI initialization failed")
            self.news_api = None
            
        try:
            self.reddit_client = self._setup_reddit()
        except:
            print("Warning: Reddit API initialization failed")
            self.reddit_client = None

    def _setup_reddit(self):
        """Setup Reddit API client"""
        return praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent="StockAnalyzer/1.0"
        )

    def fetch_stock_data(self, period="1y"):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(self.stock_symbol)
            df = stock.history(period=period)
      
            df.index = df.index.tz_localize(None)
            print(f"Fetched {len(df)} days of stock data")
            return df
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()

    def fetch_news_sentiment(self, days=7):
        """Fetch and analyze news articles"""
        if not self.news_api:
            return pd.DataFrame()
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            articles = self.news_api.get_everything(
                q=self.stock_symbol,
                from_param=from_date,
                language='en',
                sort_by='publishedAt'
            )
            
            news_data = []
            for article in articles['articles']:
                if article['title'] and article['description']:
                    text = f"{article['title']} {article['description']}"
                    clean_text = self._clean_text(text)
                    sentiment = TextBlob(clean_text).sentiment
                    
                    news_data.append({
                        'date': pd.to_datetime(article['publishedAt']).date(),
                        'text': clean_text,
                        'sentiment_polarity': sentiment.polarity,
                        'sentiment_subjectivity': sentiment.subjectivity,
                        'source': article['source']['name']
                    })
            
            return pd.DataFrame(news_data)
        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()

    def fetch_reddit_sentiment(self, days=7):
        """Fetch and analyze Reddit posts"""
        if not self.reddit_client:
            return pd.DataFrame()
        
        try:
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            posts_data = []
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                for post in subreddit.search(self.stock_symbol, time_filter='week', limit=100):
                    text = f"{post.title} {post.selftext}"
                    clean_text = self._clean_text(text)
                    sentiment = TextBlob(clean_text).sentiment
                    
                    posts_data.append({
                        'date': pd.to_datetime(datetime.fromtimestamp(post.created_utc)).date(),
                        'text': clean_text,
                        'sentiment_polarity': sentiment.polarity,
                        'sentiment_subjectivity': sentiment.subjectivity,
                        'score': post.score,
                        'num_comments': post.num_comments
                    })
            
            return pd.DataFrame(posts_data)
        except Exception as e:
            print(f"Error fetching Reddit data: {e}")
            return pd.DataFrame()

    def fetch_combined_sentiment(self, days=7):
        """Combine sentiment from multiple sources"""
        sentiment_data = []
    
        news_df = self.fetch_news_sentiment(days)
        if not news_df.empty:
            news_df['source_type'] = 'news'
            sentiment_data.append(news_df)
        
      
        reddit_df = self.fetch_reddit_sentiment(days)
        if not reddit_df.empty:
            reddit_df['source_type'] = 'reddit'
            sentiment_data.append(reddit_df)
        
        if sentiment_data:
            combined_df = pd.concat(sentiment_data, ignore_index=True)
            

            combined_df['date'] = pd.to_datetime(combined_df['date'])
            

            daily_sentiment = combined_df.groupby('date').agg({
                'sentiment_polarity': ['mean', 'std'],
                'sentiment_subjectivity': 'mean',
                'text': 'count'
            }).reset_index()
            

            daily_sentiment.columns = [
                'date',
                'avg_sentiment',
                'sentiment_std',
                'subjectivity',
                'mention_count'
            ]
            
            return daily_sentiment
        else:
            return pd.DataFrame()

    def _clean_text(self, text):
        """Clean text data"""
        if pd.isna(text):
            return ""
        
 
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        

        text = re.sub(r'[^\w\s]', '', text)
        

        text = ' '.join(text.split())
        
        return text.strip().lower()

    def _calculate_technical_indicators(self, df):
        """Calculate technical indicators"""

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
 
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        return df

    def prepare_features(self, stock_data, sentiment_data):
        """Prepare features for the model"""
        # Add technical indicators
        df = self._calculate_technical_indicators(stock_data.copy())
        

        if sentiment_data is not None and not sentiment_data.empty:
          
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            sentiment_data.set_index('date', inplace=True)
            
          
            sentiment_data.index = sentiment_data.index.tz_localize(None)
            

            df = df.join(sentiment_data, how='left')
            
          
            sentiment_columns = ['avg_sentiment', 'sentiment_std', 'subjectivity', 'mention_count']
            df[sentiment_columns] = df[sentiment_columns].fillna(0)
        

        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        

        df = df.dropna()
        
        return df

    def train_model(self, features, target, test_size=0.2):
        """Train the Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, shuffle=False
        )
        
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        

        train_score = r2_score(y_train, model.predict(X_train))
        test_score = r2_score(y_test, model.predict(X_test))
        
        return model, train_score, test_score, X_test, y_test

    def plot_predictions(self, y_true, y_pred, dates):
        """Plot actual vs predicted prices"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_true,
            name="Actual",
            line=dict(color="blue")
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=y_pred,
            name="Predicted",
            line=dict(color="red")
        ))
        
        fig.update_layout(
            title=f"{self.stock_symbol} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark"
        )
        
        return fig

# def main():
    
#     stock_symbol = "ADANIPOWER.NS"  
#     predictor = StockPredictor(stock_symbol)
    

#     stock_df = predictor.fetch_stock_data()
#     if stock_df.empty:
#         print("Failed to fetch stock data. Exiting...")
#         return
    
#     print("\nFetching sentiment data...")
#     sentiment_df = predictor.fetch_combined_sentiment()
#     if sentiment_df.empty:
#         print("No sentiment data available. Using only technical indicators...")
#     else:
#         print(f"Fetched sentiment data for {len(sentiment_df)} days")
    
#     print("\nPreparing features...")
#     feature_df = predictor.prepare_features(stock_df, sentiment_df)
#     print(f"Final dataset shape: {feature_df.shape}")

#     feature_columns = [col for col in feature_df.columns 
#                       if col not in ['Close', 'Returns', 'Date', 'Dividends', 'Stock Splits']]
#     print(f"\nUsing features: {feature_columns}")
    
#     target = feature_df['Close']

#     print("\nTraining model...")
#     model, train_score, test_score, X_test, y_test = predictor.train_model(
#         feature_df[feature_columns], 
#         target
#     )
    
#     print(f"\nModel Performance:")
#     print(f"Train Score: {train_score:.4f}")
#     print(f"Test Score: {test_score:.4f}")
    
 
#     print("\nMaking predictions...")
#     predictions = model.predict(X_test)
    

#     print("\nGenerating plot...")
#     fig = predictor.plot_predictions(
#         y_test,
#         predictions,
#         X_test.index
#     )
#     fig.show()

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"\nAn error occurred: {str(e)}")
#         import traceback
#         print(traceback.format_exc())

def main():
    # Initialize predictor
    stock_symbol = "ADANIPOWER.NS"  
    predictor = StockPredictor(stock_symbol)
    
    # Fetch stock data
    stock_df = predictor.fetch_stock_data()
    if stock_df.empty:
        print("Failed to fetch stock data. Exiting...")
        return
    
    print("\nFetching sentiment data...")
    sentiment_df = predictor.fetch_combined_sentiment()
    if sentiment_df.empty:
        print("No sentiment data available. Using only technical indicators...")
    else:
        print(f"Fetched sentiment data for {len(sentiment_df)} days")
    
    print("\nPreparing features...")
    feature_df = predictor.prepare_features(stock_df, sentiment_df)
    print(f"Final dataset shape: {feature_df.shape}")

    feature_columns = [col for col in feature_df.columns 
                      if col not in ['Close', 'Returns', 'Date', 'Dividends', 'Stock Splits']]
    print(f"\nUsing features: {feature_columns}")
    
    target = feature_df['Close']

    print("\nTraining model...")
    model, train_score, test_score, X_test, y_test = predictor.train_model(
        feature_df[feature_columns], 
        target
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = 100 * (1 - np.mean(np.abs((y_test - predictions) / y_test)))
    
    print("\n=== Model Performance Metrics ===")
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"Train Score (R²): {train_score:.4f}")
    print(f"Test Score (R²): {test_score:.4f}")

    # Generate plot
    print("\nGenerating plot...")
    fig = predictor.plot_predictions(
        y_test,
        predictions,
        X_test.index
    )
    fig.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())