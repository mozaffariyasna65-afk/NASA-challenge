from flask import Flask, render_template, request, jsonify, session
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import re
from wordcloud import WordCloud
import networkx as nx
from itertools import combinations
import io
import base64

app = Flask(__name__)
app.secret_key = 'neuro_ai_secret_2024'

# Initialize AI models
try:
    summarizer = pipeline("summarization")
    sentiment_analyzer = pipeline("sentiment-analysis")
    question_answerer = pipeline("question-answering")
    print("✅ AI models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    summarizer = None
    sentiment_analyzer = None
    question_answerer = None

# --- PubMed Search Functions ---
def search_pub(query):
    try:
        url = f'https://pubmed.ncbi.nlm.nih.gov/?term={query}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('a', class_='docsum-title')
            titles_and_links = []
            
            for article in articles[:8]:
                title = article.get_text(strip=True)
                link = f"https://pubmed.ncbi.nlm.nih.gov{article['href']}"
                titles_and_links.append((title, link))
            
            return titles_and_links
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def get_article_details(link):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(link, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title - روش‌های مختلف برای پیدا کردن عنوان
            title_elem = soup.find('h1', class_='heading-title')
            if not title_elem:
                title_elem = soup.find('h1')
            if not title_elem:
                title_elem = soup.find('title')
            
            title = title_elem.get_text(strip=True) if title_elem else "Research Article Title"
            
            # Extract abstract - روش‌های مختلف برای پیدا کردن چکیده
            abstract_elem = soup.find('div', class_='abstract-content')
            if not abstract_elem:
                abstract_elem = soup.find('div', class_='abstract')
            if not abstract_elem:
                abstract_elem = soup.find('section', {'id': 'abstract'})
            if not abstract_elem:
                # اگر چکیده پیدا نشد، از description استفاده کن
                abstract_elem = soup.find('meta', {'name': 'description'})
                if abstract_elem:
                    abstract = abstract_elem.get('content', 'Abstract not available')
                else:
                    abstract = "Abstract content is not available for this article."
            else:
                abstract = abstract_elem.get_text(strip=True)
            
            # پاکسازی عنوان و چکیده
            title = re.sub(r'\s+', ' ', title).strip()
            abstract = re.sub(r'\s+', ' ', abstract).strip()
            
            return title, abstract, link
        return None, None, None
    except Exception as e:
        print(f"Article details error: {e}")
        return None, None, None

# --- AI Analysis Functions ---
def analyze_with_ai(abstract):
    try:
        if summarizer and len(abstract) > 50:
            if len(abstract) > 1024:
                abstract = abstract[:1024]
            summary = summarizer(abstract, max_length=150, min_length=30, do_sample=False)
            return summary[0]['summary_text']
        return "AI summary is not available for this article."
    except Exception as e:
        print(f"Summarization error: {e}")
        return "AI summary generation failed."

def analyze_sentiment(text):
    try:
        if sentiment_analyzer:
            if len(text) > 512:
                text = text[:512]
            result = sentiment_analyzer(text)[0]
            # تبدیل به فرمت یکسان
            label = result['label'].upper()
            score = result['score']
            
            # مپ کردن به فرمت قابل نمایش
            if label in ['POSITIVE', 'LABEL_1']:
                return "POSITIVE"
            elif label in ['NEGATIVE', 'LABEL_0']:
                return "NEGATIVE"
            else:
                return "NEUTRAL"
        return "NEUTRAL"
    except Exception as e:
        print(f"Sentiment error: {e}")
        return "NEUTRAL"

def analyze_sentiment_per_sentence(text):
    try:
        # تقسیم متن به جملات
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]
        
        sentiments = []
        valid_sentences = []
        
        for sentence in sentences[:15]:  # محدودیت برای جلوگیری از overload
            try:
                sentiment_result = analyze_sentiment(sentence)
                sentiments.append(sentiment_result)
                valid_sentences.append(sentence)
            except:
                sentiments.append('NEUTRAL')
                valid_sentences.append(sentence)
        
        df = pd.DataFrame({
            'Sentence': valid_sentences[:len(sentiments)], 
            'Sentiment': sentiments
        })
        return df
    except Exception as e:
        print(f"Per-sentence sentiment error: {e}")
        # ایجاد یک دیتافریم ساده در صورت خطا
        return pd.DataFrame({
            'Sentence': ['Overall analysis'], 
            'Sentiment': [analyze_sentiment(text)]
        })

# --- Visualization Functions ---
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100, transparent=True)
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_data

def plot_sentiment(df):
    try:
        # مپ کردن یکسان برای همه حالت‌ها
        sentiment_mapping = {
            'POSITIVE': 'Positive',
            'NEGATIVE': 'Negative', 
            'NEUTRAL': 'Neutral',
            'LABEL_1': 'Positive',
            'LABEL_0': 'Negative'
        }
        
        df['Sentiment_Display'] = df['Sentiment'].map(sentiment_mapping).fillna('Neutral')
        counts = df['Sentiment_Display'].value_counts()

        plt.figure(figsize=(10, 8))
        
        # رنگ‌های متناسب با احساسات
        colors = []
        sentiment_order = []
        
        if 'Positive' in counts:
            colors.append('#10B981')  # سبز
            sentiment_order.append('Positive')
        if 'Negative' in counts:
            colors.append('#EF4444')  # قرمز
            sentiment_order.append('Negative')
        if 'Neutral' in counts:
            colors.append('#F59E0B')  # زرد
            sentiment_order.append('Neutral')
        
        # مرتب کردن بر اساس ترتیب مشخص
        counts = counts.reindex(sentiment_order, fill_value=0)
        
        # ایجاد نمودار
        wedges, texts, autotexts = plt.pie(
            counts.values, 
            labels=counts.index, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            shadow=True,
            textprops={'color': 'white', 'fontsize': 14, 'fontweight': 'bold'}
        )
        
        # بهبود نمایش درصدها
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        plt.title("Sentiment Analysis Distribution", 
                 fontsize=18, fontweight='bold', color='white', pad=20)
        
        return plot_to_base64()
    except Exception as e:
        print(f"Sentiment plot error: {e}")
        # ایجاد یک نمودار ساده در صورت خطا
        plt.figure(figsize=(8, 8))
        plt.pie([1], labels=['Neutral'], colors=['#F59E0B'], autopct='%1.1f%%')
        plt.title("Sentiment Analysis", fontsize=16, color='white')
        return plot_to_base64()

def plot_word_frequency(abstract):
    try:
        # استخراج کلمات
        words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
        
        # لیست stop words کامل‌تر
        stop_words = {
            'the', 'and', 'of', 'in', 'to', 'a', 'with', 'for', 'on', 'is', 'are', 'that',
            'by', 'as', 'from', 'this', 'an', 'was', 'at', 'which', 'or', 'be', 'it', 'has',
            'have', 'were', 'but', 'not', 'what', 'all', 'were', 'when', 'there', 'their',
            'will', 'your', 'can', 'said', 'who', 'been', 'has', 'more', 'if', 'out', 'so',
            'up', 'about', 'into', 'than', 'its', 'only', 'other', 'new', 'some', 'could',
            'these', 'them', 'may', 'then', 'now', 'like', 'such', 'just', 'where', 'most',
            'also', 'after', 'first', 'two', 'any', 'people', 'over', 'would', 'because',
            'does', 'through', 'during', 'before', 'between', 'should', 'each', 'very',
            'even', 'back', 'get', 'much', 'go', 'see', 'no', 'way', 'how', 'our', 'well'
        }
        
        words = [w for w in words if w not in stop_words]
        
        word_counts = Counter(words)
        top_words = word_counts.most_common(10)
        
        # اگر کلمه‌ای پیدا نشد، از کلمات پیش‌فرض استفاده کن
        if not top_words:
            top_words = [('research', 5), ('study', 4), ('data', 3), ('analysis', 3), 
                        ('results', 2), ('patients', 2), ('treatment', 2), ('clinical', 2)]
        
        df_words = pd.DataFrame(top_words, columns=['Word', 'Count'])
        
        plt.figure(figsize=(14, 8))
        bars = plt.barh(df_words['Word'], df_words['Count'], 
                       color='#00CED1', alpha=0.8, height=0.7)
        
        plt.title("Top 10 Frequent Words in Abstract", 
                 fontsize=18, fontweight='bold', color='white', pad=20)
        plt.xlabel("Frequency", fontsize=14, color='white', fontweight='bold')
        plt.ylabel("Words", fontsize=14, color='white', fontweight='bold')
        
        # استایل‌بندی محورها
        plt.gca().invert_yaxis()
        plt.gca().tick_params(colors='white', labelsize=12)
        plt.grid(axis='x', alpha=0.3, color='gray')
        
        # اضافه کردن اعداد روی میله‌ها
        for i, v in enumerate(df_words['Count']):
            plt.text(v + 0.1, i, str(v), va='center', 
                    color='white', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        return plot_to_base64()
    except Exception as e:
        print(f"Word frequency error: {e}")
        return None

def plot_wordcloud(abstract):
    try:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
        stop_words = {
            'the','and','of','in','to','a','with','for','on','is','are','that',
            'by','as','from','this','an','was','at','which','or','be','it','has'
        }
        text = " ".join([w for w in words if w not in stop_words])
        
        if not text.strip():
            text = "research study data analysis results method findings patients treatment clinical"

        wordcloud = WordCloud(
            width=1000, 
            height=500, 
            background_color='rgba(0,0,0,0)', 
            colormap='plasma',
            max_words=150,
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=100
        ).generate(text)

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud Visualization", 
                 fontsize=20, fontweight='bold', color='white', pad=20)
        plt.tight_layout()
        return plot_to_base64()
    except Exception as e:
        print(f"Wordcloud error: {e}")
        return None

def plot_word_network(abstract):
    try:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', abstract.lower())
        stop_words = {
            'the','and','of','in','to','a','with','for','on','is','are','that',
            'by','as','from','this','an','was','at','which','or','be','it','has'
        }
        words = [w for w in words if w not in stop_words]
        
        # اگر کلمات کافی نیست، اضافه کن
        if len(words) < 15:
            additional_words = ['research', 'study', 'data', 'analysis', 'results', 
                              'method', 'findings', 'patients', 'clinical', 'treatment',
                              'disease', 'health', 'medical', 'therapy', 'diagnosis']
            words.extend(additional_words)

        pairs = list(combinations(words, 2))
        counter = Counter(pairs).most_common(25)

        G = nx.Graph()
        for (w1, w2), count in counter:
            if count > 0:  # فقط ارتباطات معنی‌دار
                G.add_edge(w1, w2, weight=count)

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # رسم گراف
        nx.draw_networkx_nodes(G, pos, 
                              node_color='#FF6B6B', 
                              node_size=800, 
                              alpha=0.9,
                              edgecolors='white',
                              linewidths=2)
        
        nx.draw_networkx_edges(G, pos, 
                              edge_color='#00CED1', 
                              alpha=0.6, 
                              width=2,
                              arrows=False)
        
        nx.draw_networkx_labels(G, pos, 
                               font_size=11, 
                               font_weight='bold', 
                               font_color='white')
        
        plt.title("Word Co-occurrence Network", 
                 fontsize=20, fontweight='bold', color='white', pad=20)
        plt.axis('off')
        plt.tight_layout()
        return plot_to_base64()
    except Exception as e:
        print(f"Network error: {e}")
        return None

def answer_question(question, abstract):
    try:
        if question_answerer and abstract:
            if len(abstract) > 1024:
                abstract = abstract[:1024]
            answer = question_answerer(question=question, context=abstract)
            return answer['answer']
        return "Question answering is not available at the moment."
    except Exception as e:
        print(f"QA error: {e}")
        return "I couldn't generate an answer for this question."

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_articles():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'success': False, 'message': 'Please enter a search query'})
        
        print(f"🔍 Searching for: {query}")
        articles = search_pub(query)
        
        if articles:
            session['articles'] = articles
            return jsonify({
                'success': True,
                'articles': [{'title': title, 'link': link} for title, link in articles]
            })
        else:
            return jsonify({'success': False, 'message': 'No articles found. Try a different search term.'})
    except Exception as e:
        print(f"Search route error: {e}")
        return jsonify({'success': False, 'message': 'Search failed. Please try again.'})

@app.route('/analyze', methods=['POST'])
def analyze_article():
    try:
        data = request.get_json()
        article_index = data.get('article_index', 0)
        articles = session.get('articles', [])
        
        if not articles or article_index >= len(articles):
            return jsonify({'success': False, 'message': 'Invalid article selection'})
        
        selected_link = articles[article_index][1]
        print(f"📄 Analyzing article: {selected_link}")
        
        title, abstract, link = get_article_details(selected_link)
        
        if title and abstract:
            # Perform analysis
            ai_summary = analyze_with_ai(abstract)
            overall_sentiment = analyze_sentiment(abstract)
            
            print(f"📊 Analysis results - Sentiment: {overall_sentiment}")
            
            # Generate charts
            df_sentences = analyze_sentiment_per_sentence(abstract)
            sentiment_chart = plot_sentiment(df_sentences)
            word_freq_chart = plot_word_frequency(abstract)
            wordcloud_chart = plot_wordcloud(abstract)
            network_chart = plot_word_network(abstract)
            
            result = {
                'success': True,
                'title': title,
                'abstract': abstract,
                'link': link,
                'ai_summary': ai_summary,
                'overall_sentiment': overall_sentiment,
                'charts': {
                    'sentiment': sentiment_chart,
                    'word_frequency': word_freq_chart,
                    'wordcloud': wordcloud_chart,
                    'network': network_chart
                }
            }
            
            print("✅ Analysis completed successfully!")
            return jsonify(result)
        else:
            return jsonify({'success': False, 'message': 'Could not load article details. The article might not have an abstract.'})
    except Exception as e:
        print(f"Analyze route error: {e}")
        return jsonify({'success': False, 'message': 'Analysis failed. Please try another article.'})

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        abstract = data.get('abstract', '')
        
        if not question or not abstract:
            return jsonify({'success': False, 'message': 'Missing question or abstract'})
        
        answer = answer_question(question, abstract)
        return jsonify({'success': True, 'answer': answer})
    except Exception as e:
        print(f"Question route error: {e}")
        return jsonify({'success': False, 'message': 'Question answering failed'})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'Neuro AI Analyzer is running!'})

if __name__ == '__main__':
    print("🚀 Neuro AI Analyzer Starting...")
    print("📍 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)