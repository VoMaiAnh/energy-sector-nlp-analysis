#---------------- Librerías utilizadas  ----------------#
import json
import re
from datetime import datetime, timedelta
import calendar
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Web/Scraping
import requests
from apify_client import ApifyClient

# NLP & Text Processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import spacy
import pysentimiento
from pysentimiento.preprocessing import preprocess_tweet

# ML & Topic Modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from hdbscan import HDBSCAN
from bertopic import BERTopic

# Transformers
from transformers import pipeline


#---------------- Descarga de datos ----------------#
### Función para descargar las reseñas de Google Play
def save_GGPlay_reviews():
    # Diccionario de empresas interesadas y su ID de aplicación en Google Play Store
    empresas = {
        'Iberdrola': 'com.iberdrola.clientes',
        'Endesa': 'es.awg.movilidadEOL',
        'Naturgy': 'com.gnf.apm',
        'TotalEnergies': 'es.clientes.portal.mobile',
        'Repsol': 'com.repsol.vivit'
    }

    # Una lista vacía para guardar todas las reseñas de las empresas
    all_data = []

    # Bucle para iterar cada empresa y obtener las reseñas
    for empresa, appID in empresas.items():
        result = reviews_all(
            app_id=appID,
            lang='es',  
            country='es',  
            sort=Sort.MOST_RELEVANT,  
            sleep_milliseconds=0
        )

        df = pd.DataFrame(result)

        # Añadir la columna 'Company' para guardar nombre de la empresa
        df['Company'] = empresa 

        ### Transformar la columna datetime al formato adecuado
        # Convertir las columnas 'at' y 'repliedAt' a datetime
        df['at'] = pd.to_datetime(df['at'])
        df['repliedAt'] = pd.to_datetime(df['repliedAt'])

        df['at'] = df['at'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        df['repliedAt'] = df['repliedAt'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Exportar el DataFrame de cada empresa a un archivo JSON
        file_name = f'{empresa}_es_18Ene2025.json'
        df.to_json(file_name, orient='records', lines=True, force_ascii=False)
        print(f"Se ha guardado {len(df)} reseñas de {empresa} en formato json en el fichero {file_name}")
        
        # Añadir el DataFrame a la lista combinada de reseñas
        all_data.append(df)

    # Combine all DataFrames into one after the loop
    combined_df = pd.concat(all_data, ignore_index=True)

    # Exportar el DataFrame de cada empresa a un archivo JSON
    file_name = f'GGPlay_es_18Ene2025.json'
    combined_df.to_json(file_name, orient='records', lines=True, force_ascii=False)
    print(f"Se ha guardado {len(combined_df)} reseñas de todas las empresas en formato json en el fichero {file_name}")
    return combined_df

### Funciones para descargar los Tweets
apify_client = ApifyClient(
    token=os.getenv('APIFY_API'),
    max_retries=8,
    min_delay_between_retries_millis=500, # 0.5s
    timeout_secs=600, # 10 mins
)

# Función para crear timestamp
def generate_monthly_date_range(start_year):
    # Start from the beginning of the year
    start_date = datetime(start_year, 1, 1)
    
    # Generate date ranges for each month in the year
    date_ranges = []
    
    for month in range(1, 13):  # Loop through 12 months
        # Get the start date of the month
        start_str = start_date.strftime("%Y-%m-%d")
        
        # Get the last day of the current month
        _, last_day = calendar.monthrange(start_date.year, start_date.month)
        end_date = datetime(start_date.year, start_date.month, last_day) + timedelta(days=1)
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Create the timestamp string
        timestamp = f"until:{end_str} since:{start_str}"
        
        # Add the string to the list
        date_ranges.append(timestamp)
        
        # Move to the next month
        # The next start date will be the first day of the next month
        start_date = datetime(start_date.year, start_date.month, 1) + timedelta(days=32)
        start_date = datetime(start_date.year, start_date.month, 1)  # Normalize to the first of the next month
    
    return date_ranges


# Función para buscar y guardar los Tweets    
def save_Tweets(company, search_by_company, year):
    date_ranges = generate_monthly_date_range(year)
    lang_query = " lang:es "

    for timestamp in date_ranges:
        search_term = search_by_company + lang_query + timestamp
        start_date = timestamp.split(' since:')[1].split(' until:')[0]  # Extract the start date
        month = start_date.split('-')[1] 
        print(f"Término de búsqueda para Tweets de {company} en mes {month} de {year}: {search_term}")

        run_input = {
            "searchTerms": [search_term],
            "sort": "Latest",
            "maxItems": 10000
        }
        run = apify_client.actor("nfp1fpt5gUlBwPcor").call(run_input=run_input)

        datasetID = run["defaultDatasetId"]
        url = f"https://api.apify.com/v2/datasets/{datasetID}/items?token={os.getenv('APIFY_API')}"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"Número de Tweets obtenidos de {company} en mes {month} de {year}: {len(data)}")
        else:
            print(f"Error: {response.status_code}")

        # File path
        file_path = f"TFM/Twitter/apify_{company}_{month}_{year}.json"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        
        print(f"Se han guardado {len(data)} tweets de {company} en el fichero {file_path}")
        print(f"\n----------------------------")

# Endesa
query = "-baloncesto, -LigaEndesa, -Liga, -partido (@EndesaClientes OR @Endesa)"
save_Tweets("Endesa",query, 2024)
save_Tweets("Endesa",query, 2023)
# Iberdrola
query = "-Copa, -CopaDeLaReina, -goles, -gol, -fútbol, -deporte, -partido (@TuIberdrola OR @iberdrola)"
save_Tweets("Iberdrola",query, 2024)
save_Tweets("Iberdrola",query, 2023)
# Naturgy
query = "(@NaturgyClientEs OR @Naturgy)"
save_Tweets("Naturgy",query, 2024)
save_Tweets("Naturgy",query, 2023)
# Repsol
query = "@Repsol"
save_Tweets("Repsol",query, 2024)
save_Tweets("Repsol",query, 2023)
# TotalEnergies
query = "-correr -maratón -pie -maratones -deporte (@TotalEnergiesES)"
save_Tweets("TotalEnergies",query, 2024)
save_Tweets("TotalEnergies",query, 2023)

# Función para combinar Tweets en un fichero de datos 
def combine_Tweets_all(company,year):
    df_tweets_all = pd.DataFrame(columns=["company","tweet", "author","url","createdAt", "lang", "location", "likeCount", "quoteCount","retweetCount", "replyCount","viewCount","media"])

    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    for month in months:

        file_path = f"TFM/Twitter/apify_{company}_{month}_{year}.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract and append data to the DataFrame
        for content in data:
            # Safely extract the media URL
            media_url = None  # Default value if media URL is not available
            entities = content.get("entities", {})
            media = entities.get("media", [])
            if media and isinstance(media, list):  # Check if media exists and is a list
                media_url = media[0].get("media_url_https") if len(media) > 0 else None
            result = {
                "company": company,
                "tweet": content.get("fullText"),
                "author": content["author"].get("name"),
                "url": content.get("url"),
                "createdAt": content.get("createdAt"),
                "lang": content.get("lang"),
                "location": content["author"].get("location"),
                "likeCount": content.get("likeCount"),
                "quoteCount": content.get("quoteCount"),
                "retweetCount": content.get("retweetCount"),
                "replyCount": content.get("replyCount"),
                "viewCount": content.get("viewCount"),
                "media": media_url,     
            }
            df_tweets_all = pd.concat([df_tweets_all, pd.DataFrame([result])], ignore_index=True)


        print(f"Se han añadido {len(data)} tweets de {company} del mes {month} de {year}")

        # Exportar el DataFrame de todas las empresas a un archivo JSON
    file_name = f'TFM/Twitter/{company}_{year}_Tweets.json'
    df_tweets_all.to_json(file_name, orient='records', lines=True, force_ascii=False)
    print(f"Se han guardado en total {len(df_tweets_all)} tweets de {company} de {year} en el fichero {file_name}") 

empresas = ['Iberdrola','Endesa','Naturgy','TotalEnergies','Repsol']
Tweets_all = []
for empresa in empresas:
    file_path = f"TFM/Twitter/{empresa}_2024_Tweets.json"
    df = pd.read_json(file_path, orient='records', lines=True)
    Tweets_all.append(df)
    print(f"Se han añadido {len(df)} tweets de {empresa} en 2024 a Tweets_all")
    print(f"\n----------------------------")

for empresa in empresas:
    file_path = f"TFM/Twitter/{empresa}_2023_Tweets.json"
    df = pd.read_json(file_path, orient='records', lines=True)
    Tweets_all.append(df)
    print(f"Se han añadido {len(df)} tweets de {empresa} en 2023 a Tweets_all")
    print(f"\n----------------------------")

Tweets_df = pd.concat(Tweets_all, ignore_index=True)
file_name = f'TFM/Twitter/Tweets_es_18Ene2025.json'
Tweets_df.to_json(file_name, orient='records', lines=True, force_ascii=False)
print(f"Se ha guardado {len(Tweets_df)} tweets de todas las empresas en formato json en el fichero {file_name}")

#---------------- Preprocesamiento ----------------#
### Función para transformar datos de fechas y horas en formato correcto
def convert_datetime(value):
    try:
        # Handle numeric or string Unix timestamps
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            # Decide between milliseconds and seconds
            if len(str(int(value))) == 13:  # Milliseconds
                return pd.to_datetime(int(value), unit='ms')
            elif len(str(int(value))) == 10:  # Seconds
                return pd.to_datetime(int(value), unit='s')
        # Handle standard ISO 8601 strings or datetime strings
        return pd.to_datetime(value)
    except Exception as e:
        # Return NaT for invalid formats
        return pd.NaT
df_gg['timestamp'] = df_gg['at'].apply(convert_datetime)

# Datos faltantes
print(f"Missing Values: {df_gg.isnull().sum()}")
# Datos duplicados
duplicates = GG_app_dict.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")
Tweets = Tweets.drop_duplicates()
print("Number of observation after removing duplicates: " + str(len(Tweets)))

### Función para limpiar texto: eliminar menciones, URL, caracteres especiales y espacios sobrantes.
def preprocess_text_sentiment(text):
    # Lowercase the text
    text = text.lower()
    # Remove @ symbol
    text = re.sub(r"@", "", text)  

    text = preprocess_tweet(text, shorten=2, lang="es") # Treat URL, shorten repeated characters, normaliza laughters, handles emojis

    # Elimina caracteres especiales pero mantiene letras, números, espacios y acentos
    text = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#---------------- EDA ----------------#
df_gg.info()
df_gg.describe()

### Gráfico de número de reseñas/tweets por año
# Número de comentarios por año
df_gg['year'] = df_gg['timestamp'].dt.year
comments_by_year = df_gg.groupby('year').size().reset_index(name='count')

# Gráfico de líneas
plt.figure(figsize=(10, 5))
sns.lineplot(data=comments_by_year, x='year', y='count', marker='o', color='steelblue')
plt.title('Número de reseñas total por año', fontsize=14)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Número de reseñas', fontsize=12)
plt.xticks(comments_by_year['year'].unique())
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

### Gráfico de Distribución de Reseñas/Tweets por empresa
# Número de comentarios/tweets por empresa ordenados por porcentaje de forma descendente
company_counts = df_gg['Company'].value_counts()
company_counts = company_counts.sort_values(ascending=True)
total_comments = company_counts.sum()

# Colores customizados por empresa
custom_colors = {
    'Iberdrola': '#4CAF50',     # Medium green
    'Endesa': '#42A5F5',        # Medium blue
    'Naturgy': '#FFA726',       # Medium orange
    'Repsol': '#EF5350',        # Medium red
    'TotalEnergies': '#9966CC'  # Medium purple
}
colors = [custom_colors.get(company, 'gray') for company in company_counts.index]

# Gráfico de torta
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    company_counts.values,
    # labels=company_counts.index,
    autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
    colors=colors,
    startangle=90,
    wedgeprops={'edgecolor': 'white'}
)

# Añadir leyenda
ax.legend(
    wedges,
    company_counts.index,
    title="Empresa",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)
plt.title('Distribución de Reseñas por Empresa', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()

### Gráfico de Evolución de Reseñas/Tweets por empresa a lo largo del tiempo
# Colores customizados por empresa
custom_colors = {
    'Iberdrola': '#4CAF50',     # Medium green
    'Endesa': '#42A5F5',        # Medium blue
    'Naturgy': '#FFA726',       # Medium orange
    'Repsol': '#EF5350',        # Medium red
    'TotalEnergies': '#9966CC'  # Medium yellow
}

# Número de comentarios/tweets por empresa y año
df_gg['year'] = df_gg['timestamp'].dt.year
comments_by_year_company = df_gg.groupby(['year', 'Company']).size().reset_index(name='count')

# Gráfico de línea
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=comments_by_year_company,
    x='year',
    y='count',
    hue='Company',
    marker='o',
    palette=custom_colors  # Use custom palette here
)
plt.title('Number of Comments by Year and Company', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Comments', fontsize=12)
plt.xticks(comments_by_year_company['year'].unique())
plt.legend(title='Company', loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

### Gráfico de Distribución de número de palabras por cada reseña/tweet
df_gg['review_length'] = df_gg['content'].apply(lambda x: len(x.split()))

# Histograma
plt.figure(figsize=(10, 6))
sns.histplot(df_gg['review_length'], bins=50, kde=True, color='skyblue')
plt.title('Distribución de número de palabras en las reseñas', fontsize=14)
plt.xlabel('Número de palabras', fontsize=12)
plt.ylabel('Número de reseñas', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

#---------------- Análisis de sentimiento ----------------#
### Con "nlptown/bert-base-multilingual-uncased-sentiment" para reseñas de Google Play
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_sentiment(tweet):
    results = sentiment_pipeline(tweet, return_all_scores=True)[0]
    
    negative = sum(item["score"] for item in results if item["label"] in ["1 star", "2 stars"])
    neutral  = sum(item["score"] for item in results if item["label"] == "3 stars")
    positive = sum(item["score"] for item in results if item["label"] in ["4 stars", "5 stars"])
    
    # Get the overall classification by taking the label with the highest score
    overall_label = max(results, key=lambda x: x["score"])["label"]
    
    # Convert scores to percentages (rounded to, e.g., 2 decimal places)
    pos_pct = round(positive * 100, 2)
    neu_pct = round(neutral * 100, 2)
    neg_pct = round(negative * 100, 2)
    
    return overall_label, pos_pct, neu_pct, neg_pct

# Apply the function to your dataframe and expand the results into new columns
df_gg[['Sentiment_NLPTown', 'Positive_pct', 'Neutral_pct', 'Negative_pct']] = df_gg['cleaned_content'].apply(lambda x: pd.Series(get_sentiment(x)))

print(df_gg[['Sentiment_NLPTown', 'Positive_pct', 'Neutral_pct', 'Negative_pct']].head())

def classify_sentiment(pos, neu, neg, threshold=50, margin=15):
    top = sorted([("POS", pos), ("NEU", neu), ("NEG", neg)], key=lambda x: -x[1])
    if top[0][1] < threshold:
        return "MIXED"
    if abs(top[0][1] - top[1][1]) < margin:
        return "MIXED"
    return top[0][0]

df_gg['Sentiment'] = df_gg.apply(
    lambda row: classify_sentiment(row['Positive_pct'], row['Neutral_pct'], row['Negative_pct']),
    axis=1
)

### Con RoBertuito de Pysentimiento para Tweets
# Initialize the sentiment analyzer for Spanish
analyzer = create_analyzer(task="sentiment", lang="es")

# Define the sentiment analysis function
def analyze_sentiment(text):
    """
    Perform sentiment analysis on the given text.

    Args:
        text (str): The input text in Spanish.

    Returns:
        dict: A dictionary with sentiment label and probabilities.
    """
    if not text or not isinstance(text, str):
        return {"sentiment": None, "NEG": None, "NEU": None, "POS": None}
    
    result = analyzer.predict(text)
    return {
        "sentiment": result.output,  # Sentiment label
        "NEG": result.probas.get("NEG", None)*100,
        "NEU": result.probas.get("NEU", None)*100,
        "POS": result.probas.get("POS", None)*100
    }


# Apply the sentiment analysis function to the column
Tweet_df["pysentimiento"] = Tweet_df["cleaned_tweet"].apply(analyze_sentiment)

# Extract sentiment and probabilities into separate columns
Tweet_df["sentiment_pysentimiento"] = Tweet_df["pysentimiento"].apply(lambda x: x["sentiment"])
Tweet_df["NEG"] = Tweet_df["pysentimiento"].apply(lambda x: x["NEG"])
Tweet_df["NEU"] = Tweet_df["pysentimiento"].apply(lambda x: x["NEU"])
Tweet_df["POS"] = Tweet_df["pysentimiento"].apply(lambda x: x["POS"])

# Drop the intermediate 'sentiment_results' column if not needed
Tweet_df.drop(columns=["pysentimiento"], inplace=True)

Tweet_df['Sentiment'] = Tweet_df.apply(
    lambda row: classify_sentiment(row['POS'], row['NEU'], row['NEG']),
    axis=1
)

### Visualización de los resultados
## Gráfico de Distribución de Sentimientos en las Reseñas
# Count and sort sentiment categories by count (ascending)
sentiment_counts = df_gg['Sentiment'].value_counts().sort_values(ascending=True)

# Define moderate custom colors
color_map = {
     'POS': '#81C784',
     'NEG': '#E57373',
     'NEU': '#64B5F6',
     'MIXED': '#FFD54F'
}
colors = [color_map[label] for label in sentiment_counts.index]

# Prepare figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Draw pie chart without percentage labels
wedges, texts = ax.pie(
    sentiment_counts,
    # labels=sentiment_counts.index,
    startangle=90,
    # colors=colors,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
)

# Add percentage labels manually outside the pie with leader lines
total = sentiment_counts.sum()
for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2.0
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))

    percentage = sentiment_counts.iloc[i] / total * 100
    label = f"{percentage:.1f}%"

    # Set label position slightly outside the pie
    ax.annotate(label,
                xy=(x, y), 
                xytext=(1.2 * x, 1.2 * y),
                ha='center', va='center',
                fontsize=15,
                arrowprops=dict(arrowstyle='-', color='gray'))

# Add legend (Categoría)
ax.legend(wedges, sentiment_counts.index, title='Categoría', loc='lower left', bbox_to_anchor=(1, 0.5))

# Title and layout
plt.title('Distribución de Sentimientos en las Reseñas')
plt.axis('equal')
plt.tight_layout()
plt.show()

## Gráfico de Distribución de Sentimientos por Empresa
# Group and count sentiments by company
grouped = df_gg.groupby(['Company', 'Sentiment']).size().unstack(fill_value=0)

desired_order = ['NEG', 'POS', 'MIXED', 'NEU']
grouped = grouped.reindex(columns=desired_order)

color_map = {
    'POS': '#2ca02c',    # Standard Green
    'NEG': '#d62728',    # Standard Red
    'NEU': '#1f77b4',    # Standard Blue
    'MIXED': '#ff7f0e'   # Standard Orange
}
# Match colors to the sentiment columns in correct order
colors = [color_map.get(col, '#999999') for col in grouped.columns]
# Plot
grouped.plot(kind='bar', stacked=False, figsize=(10, 7), color=colors)
plt.title('Distribución de Sentimientos por Empresa')
plt.xlabel('Empresa')
plt.ylabel('Número de Reseñas')
plt.xticks(rotation=45)
plt.legend(title='Sentimiento')
plt.tight_layout()
plt.show()

## Gráfico de la Evolución del Sentimiento a lo Largo del Tiempo
df_gg['timestamp'] = pd.to_datetime(df_gg['timestamp'])
# Group by month and sentiment
time_sentiment = df_gg.groupby([df_gg['timestamp'].dt.to_period('M'), 'Sentiment']).size().unstack().fillna(0)

# Define custom colors for each sentiment
color_map = {
    'POS': '#81C784',    # Green
    'NEG': '#E57373',    # Red
    'NEU': '#64B5F6',    # Blue
    'MIXED': '#FFD54F'   # Yellow
}

# Match colors to the sentiment columns in correct order
colors = [color_map.get(col, '#999999') for col in time_sentiment.columns]

# Plot with custom colors
time_sentiment.plot(figsize=(12, 6), color=colors)

# Labels and formatting
plt.title('Evolución del Sentimiento a lo Largo del Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Número de Reseñas')
plt.xticks(rotation=45)
plt.legend(title='Sentimiento')
plt.tight_layout()
plt.show()

## Gráfico de Nubes de palabras destacadas (TF-IDF) de comentarios negativos y positivos
# Filtrar textos positivos y negativos
df_pos = df_gg[df_gg['Sentiment'] == 'POS']
df_neg = df_gg[df_gg['Sentiment'] == 'NEG']

# Crear corpus por grupo (un documento por grupo)
corpus = {
    'POS': ' '.join(df_pos['processed_text'].dropna().astype(str)),
    'NEG': ' '.join(df_neg['processed_text'].dropna().astype(str))
}

# TF-IDF vectorization con stopwords en español
vectorizer = TfidfVectorizer(stop_words=spanish_stopwords, max_features=100)
X = vectorizer.fit_transform(corpus.values())
feature_names = vectorizer.get_feature_names_out()

# Crear WordClouds
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for idx, (sentiment, vector) in enumerate(zip(corpus.keys(), X.toarray())):
    word_weights = dict(zip(feature_names, vector))
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='Greens' if sentiment == 'POS' else 'Reds'
                  ).generate_from_frequencies(word_weights)

    axes[idx].imshow(wc, interpolation='bilinear')
    axes[idx].set_title(f'Palabras Relevantes (TF-IDF) - {sentiment}', fontsize=14)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

## Gráfico de Relación entre Sentimiento Positivo y Negativo en Reseñas MIXED
mixed_df = df_gg[df_gg['Sentiment'] == 'MIXED']
plt.figure(figsize=(10, 6))
sns.scatterplot(data=mixed_df, x='Positive_pct', y='Negative_pct', alpha=0.6)
plt.title('Relación entre Sentimiento Positivo y Negativo en Reseñas MIXED')
plt.xlabel('Porcentaje Positivo')
plt.ylabel('Porcentaje Negativo')
plt.grid(True)
plt.tight_layout()
plt.show()

#---------------- Clasificación Zero-shot ----------------#
# Load the zero-shot classification pipeline (ensure the model supports Spanish)
classifier = pipeline(
    "zero-shot-classification",
    model="Recognai/zeroshot_selectra_medium",
    tokenizer="Recognai/zeroshot_selectra_medium",
    use_fast=True 
)

# Candidate labels for sentiment classification in Spanish
content_labels = labels = ["empresa", "aplicación"]

def classify_tweet(tweet, candidate_labels, hypothesis_template):
    result = classifier(tweet, candidate_labels, hypothesis_template=hypothesis_template)
    # Get the highest scoring label and its score
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return top_label, top_score

# Example function to classify both general and company-specific sentiment
def analyze_tweet_sentiment(tweet):
    company_template = "Este comentario está dirigido a la {}."
    company_label, company_score = classify_tweet(tweet, content_labels, hypothesis_template=company_template)

    return {
        "company_sentiment": company_label,
        "company_score": company_score
    }

import time
from tqdm import tqdm
tqdm.pandas()
results = []

for idx, row in tqdm(df_gg.iterrows(), total=len(df_gg)):
    try:
        result = analyze_tweet_sentiment(row["cleaned_content"], row["Company"])
        results.append(result)
    except Exception as e:
        results.append({"company_sentiment": None, "company_score": None})

    # Optional: Save every 100 rows
    if idx % 500 == 0:
        pd.DataFrame(results).to_csv("partial_results_Recognai.csv", index=False)
        time.sleep(0.1)  # to avoid hitting rate limits if using an API
        
## Gráfico de Distribución de Categorías de Sentimiento por Empresa 
sentiment_counts = df_gg['company_sentiment'].value_counts().sort_values(ascending=True)
colors = sns.color_palette('Set2', n_colors=len(sentiment_counts))

plt.figure(figsize=(6, 6))
plt.pie(
    sentiment_counts.values,
    labels=sentiment_counts.index,
    autopct='%1.2f%%',  # Show percentages with 2 decimal places
    startangle=90,
    colors=colors)
plt.title('Distribución de Categorías de Sentimiento por Empresa')
plt.axis('equal')
plt.tight_layout()
plt.show()

#---------------- Topic Modeling - LDA ----------------#
nltk.download('stopwords')

# Load the Spanish language model from spaCy
nlp = spacy.load("es_core_news_sm")

# Define Spanish stopwords (default NLTK + custom words)
spanish_stopwords = set(stopwords.words('spanish'))
spanish_stopwords.update((
    "iberdrola", "endesa", "repsol", "naturgy", 
    "totalenergies", "tuiberdrola", "españa", 
    "empresa", "empresas", "compañía"
))


temporal_words = {"día", "días", "mes", "meses", "año", "años", "vez", "veces", "ahora", "semana", "semanas", "hoy", "mañana", "ayer", "hora", "horas", "momento", "siempre"}

spanish_stopwords.update(temporal_words)

def preprocess_tweet(text):
    """
    Cleans, tokenizes, and lemmatizes a Spanish tweet.
    
    Steps:
      - Lowercases the text.
      - Removes URLs.
      - Removes user mentions and '#' symbols.
      - Uses spaCy to tokenize and lemmatize the text.
      - Filters out stopwords and non-alphabetic tokens.
      - Optionally, filters tokens based on desired part-of-speech.
    """
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove mentions and '#' symbols (retains the word following '#')
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    # Process the text with spaCy for tokenization and lemmatization
    doc = nlp(text)
    
    # Define valid parts-of-speech (you can adjust these as needed)
    VALID_POS = {'PROPN', 'NOUN'}
    
    # Generate a list of lemmas for tokens that are alphabetic, 
    # not in the stopwords list, and belong to valid POS.
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha 
        and token.lemma_ not in spanish_stopwords
        and token.pos_ in VALID_POS
    ]
    
    return tokens

# Assuming your DataFrame is named 'Tweet_df' and contains a column 'tweet'
# Apply the preprocessing function to the 'tweet' column
Tweet_df['processed_text'] = Tweet_df['tweet'].apply(preprocess_tweet).tolist()

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(Tweet_df['processed_text'].to_list())
# Assign ids to every word without gaps
dictionary.compactify()
# Filter out words that appear in less than 2 documents, thos appearing in more than 97% of corpus, and keep all other words
dictionary.filter_extremes(no_below=2, no_above=0.97, keep_n=None)
# Reassign ids to every word without gaps after filtering
dictionary.compactify()
# Convert document into a bag-of-words format
corpus = [dictionary.doc2bow(doc) for doc in Tweet_df['processed_text']]

import math
# ---------------------------
# Step 1: Train multiple LDA models and compute coherence
# ---------------------------
min_topics = 2
max_topics = 10  # Change as needed
coherence_values = []
lda_models = {}

for num_topics in range(min_topics, max_topics+1):
    print(f"Training LDA with {num_topics} topics...")
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
    
    # Compute Coherence Score using the 'c_v' metric
    coherence_model = CoherenceModel(model=lda_model, texts=Tweet_df['processed_text'], dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_values.append((num_topics, coherence_score))
    lda_models[num_topics] = lda_model
    print(f"Number of Topics: {num_topics} \t Coherence Score: {coherence_score:.4f}")
    
    # ---------------------------
    # Step 2: Generate WordClouds for each topic in the current model
    # ---------------------------
    # Calculate the number of rows needed (2 columns)
    num_rows = math.ceil(num_topics / 2)

    # Create subplots with a smaller overall figsize
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(8, num_rows*3))
    axes = axes.flatten()  # flatten to iterate easily

    for topic in range(num_topics):
        topic_words = dict(lda_model.show_topic(topic, topn=30))
        wordcloud = WordCloud(width=400, height=200, background_color='white')\
                    .generate_from_frequencies(topic_words)
        axes[topic].imshow(wordcloud, interpolation='bilinear')
        axes[topic].axis("off")
        axes[topic].set_title(f"LDA with {num_topics} Topics - Topic {topic}")

    # Hide any unused subplots (if num_topics is odd)
    for i in range(num_topics, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------
# Step 3: Evaluate Models
# ---------------------------
# Print coherence scores for each model
print("\nCoherence Scores for different numbers of topics:")
for num_topics, score in coherence_values:
    print(f"Topics: {num_topics} \t Coherence: {score:.4f}")

# You might choose the model with the highest coherence score and the fewest topics.
# For example, if coherence increases until 5 topics and then plateaus or decreases,
# you might select the 5-topic model.

# Optionally, plot the coherence scores to visualize the trend:
topic_nums, scores = zip(*coherence_values)
plt.figure(figsize=(8, 4))
plt.plot(topic_nums, scores, marker='o')
plt.title("LDA Model Coherence Scores")
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score (c_v)")
plt.xticks(topic_nums)
plt.show()

# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, passes=10, random_state=42)

# Print the topics
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Visualize the topics
pyLDAvis.enable_notebook()
lda_vis = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False, n_jobs=1)
# Display the visualization
pyLDAvis.display(lda_vis)

#---------------- Topic Modeling - BERTopic ----------------#
stopword_es = nltk.corpus.stopwords.words('spanish')
stopword_es.extend(("http","www","com","https","iberdrola", "endesa", "repsol", "naturgy", "totalenergies", "tuiberdrola", "empresa", "empresas", "compañía", "españa","día", "mes", "año", "hoy", "mañana", "ayer", "veces", "semana", "horas", "url", "emoji", "emojis"))
# Define token pattern that supports Spanish accented characters
regex_pattern = r'(?u)\b[\wáéíóúüñ]{3,}\b'

vectorizer_model = CountVectorizer(
    stop_words=list(stopword_es),
    min_df=0.005,       # increase minimum document frequency
    max_df=0.90,        # decrease maximum document frequency
    ngram_range=(1, 2),
    token_pattern=regex_pattern,
    dtype=np.int32,
    max_features=5000   # limit the vocabulary size
)
#reducir impacto palabras muy altas
from bertopic.vectorizers import ClassTfidfTransformer
import numpy as np
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Pre-calculate embeddings
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(Tweet_df['cleaned_tweet'], show_progress_bar=True)
embeddings = embeddings.astype("float32")

# Controlar número de topics
hdbscan_model = HDBSCAN(min_cluster_size=2500, metric='euclidean', cluster_selection_method='eom', prediction_data=True)



topic_model_new = BERTopic(language="spanish",
                           calculate_probabilities=True,
                           nr_topics = "auto",
                      vectorizer_model = vectorizer_model,
                          ctfidf_model =ctfidf_model,
                         hdbscan_model = hdbscan_model,
                          embedding_model = embedding_model)

topics_new, probs_new = topic_model_new.fit_transform(Tweet_df['cleaned_tweet'], embeddings)

# Get the top words for each topic
topics = topic_model_new.get_topics()

# Print the topics
for topic_id, words in topics.items():
    print(f"Topic {topic_id}:")
    print([word for word, _ in words], "\n")

# Plot word clouds for the new topics
new_topics = topic_model_new.get_topic_info()
for topic_num in new_topics.Topic:
    # if topic_num == -1:  # Skip outlier topic
    #     continue
    words_freq = dict(topic_model_new.get_topic(topic_num))
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Topic {topic_num + 2}")
    plt.show()

### Mapa intertopico
all_topics = sorted(set(topics_new))  # topics_new is the list returned from .fit_transform()
fig = topic_model_new.visualize_topics(topics=all_topics)
fig.show()
fig.write_html("topic_Tweet.html")

### Gráfico de baras de Distribución de palabras para cada tópico
all_topics = list(topic_model_new.get_topics().keys())
fig = topic_model_new.visualize_barchart(n_words=20, topics = all_topics, top_n_topics=None, height=1500, width=1000)
for ann in fig.layout.annotations:          # facet titles live as annotations
    ann.font.size = 30           
fig.update_layout(
        font=dict(size=30),                 # global font
        yaxis_tickfont_size = 30,
        xaxis_tickfont_size = 30,
)
fig.write_html("bar_topic.html")
fig.show()

### Gráfico jerárquico de tópicos
fig = topic_model_new.visualize_hierarchy()
fig.show()
fig.write_html("bar_hierarchy.html")

### Matriz de similaridad de tópicos
fig = topic_model_new.visualize_heatmap()
fig.show()
fig.write_html("bar_heatmap.html")