import pandas as pd
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# LOAD THE DATASET
# -----------------------------------------------------------------------------
load_dotenv()

df = pd.read_json('TFM/Tweets_company_sentiment.json',
                  orient='records',
                  lines=True)

print(f"✅ Shape: {df.shape}") # (153238, 22)
df.rename({"Sentiment":"sentiment_nlptown"}, inplace=True, axis="columns")
df.columns

df['company'].unique()
# Index(['company', 'tweet', 'author', 'url', 'createdAt', 'lang', 'location',
#        'likeCount', 'quoteCount', 'retweetCount', 'replyCount', 'viewCount',
#        'media', 'timestamp', 'cleaned_tweet', 'sentiment_pysentimiento', 'NEG',
#        'NEU', 'POS', 'sentiment_nlptown', 'company_sentiment',
#        'company_score'],
#       dtype='object')

# -----------------------------------------------------------------------------
# cONSTRUCT A SMALL RANDOM TEST SET
# -----------------------------------------------------------------------------
# 1. Define the total size and calculate per-group size
total_sample_size = 100
companies = df['company'].unique()
size_per_company = total_sample_size // len(companies)

# 2. Perform the stratified random sampling
df_test = df.groupby('company', group_keys=False).apply(
    lambda x: x.sample(n=size_per_company, random_state=42)
)

# 3. Shuffle the final dataset (optional but recommended for evaluation)
df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Verify the proportions
print(df_test['company'].value_counts())

# -----------------------------------------------------------------------------
# OPTION 1: NORMAL BATCH FUNCTION
# -----------------------------------------------------------------------------
# 1. Gemini model via Vertex AI API is used in this example but can be replaced by any other LLM
client = genai.Client(
    vertexai=True,
    api_key=os.getenv("GOOGLE_CLOUD_API_KEY"),
)

model = "gemini-3-pro-preview"

# 2. Define the functions
def analyze_tweet(tweet: str, company: str, author: str) -> dict:
    prompt = f"""Analiza este tweet de {company}:
    Tweet: {tweet}
    Autor: {author}

    Devuelve un JSON con este formato:
    {{"company": "{company}", "sentiment": "POS|NEG|NEU|MIXED", "subject": "billing|service|app|pricing|sustainability|other", "urgency": "high|medium|low", "summary": "1 frase en español"}}"""

    # Force the model to output JSON
    generate_content_config = types.GenerateContentConfig(
       temperature=0.1,
       response_mime_type="application/json", 
    )

    try:
       response = client.models.generate_content(
       model=model,
       contents=prompt,
       config=generate_content_config,
       )
       
       # With application/json, response.text IS the JSON string
       return json.loads(response.text)
        
    except Exception as e:
       print(f"Error processing tweet from {author}: {e}")
       return {
       "company": company,
       "sentiment": "NA",
       "issue": "NA",
       "urgency": "NA",
       "summary": "Error al procesar JSON"
       }
    
def analyze_batch(df: pd.DataFrame, n_tweets=None):
    results = []
    if n_tweets is not None:
       df = df.head(n_tweets)
    for _, row in df.iterrows():
       analysis = analyze_tweet(row['cleaned_tweet'], row['company'], row['author'])
       row_dict = row.to_dict()
       row_dict.update(analysis)
       results.append(row_dict)
       print(analysis)
    
    enriched_df = pd.DataFrame(results)
    enriched_df.to_json('TFM/gemini_tweets_enriched.json', orient='records', lines=True, force_ascii=False)
    return enriched_df

# 3. Execute the batch functions - UNCOMMENT THIS SECTION TO TEST THE FUNCTIONS
# result = analyze_batch(df, 100)
# print(result)


# -----------------------------------------------------------------------------
# OPTION 2: LANGCHAIN'S CUSTOMIZED SENTIMENT CHAIN
# -----------------------------------------------------------------------------
import os
import json
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import time

# 1. Initialize the LangChain LLM wrapper
# Create a service account in the Google Cloud Console of your prohject, download the json key and set the path to the key in your .env as GOOGLE_APPLICATION_CREDENTIALS
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_CLOUD_API_KEY"),
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    temperature=0.1,
    vertexai=True
)

# UNCOMMENT this to test the llm
# response = llm.invoke([HumanMessage(content="Hello! How are you?")])

# 2. Define the Chain Components
template = """Analiza este tweet de {company}:
    Tweet: {tweet}
    Autor: {author}

    Devuelve un JSON con este formato:
    {{
    "company": "{company}", 
    "sentiment": "POS|NEG|NEU|MIXED", 
    "subject": "billing|service|app|pricing|sustainability|other", 
    "urgency": "high|medium|low", 
    "summary": "1 frase en español"
    }}
    Responde solo con el JSON."""

prompt = PromptTemplate.from_template(template)
parser = JsonOutputParser()

# 3. Define the "Save to JSON" step
def save_result_to_json(data: dict):
    """Appends the result to a JSONL file immediately after generation"""
    file_path = 'TFM/gemini_tweets_enriched.json'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a', encoding='utf-8') as f:
        # force_ascii=False keeps the Spanish characters (ñ, í) readable
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')
    
    return data  # Return data so the chain can continue if needed

# 4. Create the Chain (LCEL)
# This pipes the prompt to the model and then to the JSON parser
sentiment_chain = prompt | llm | parser | RunnableLambda(save_result_to_json)

def analyze_tweet_langchain(tweet: str, company: str, author: str) -> dict:
    try:
        # Run the chain
        return sentiment_chain.invoke({
            "tweet": tweet, 
            "company": company, 
            "author": author
        })
    except Exception as e:
        print(f"LangChain Error: {e}")
        return {
            "company": company, "sentiment": "NA", "subject": "NA", 
            "urgency": "NA", "summary": "Error en LangChain"
        }

# 5. Batch function - Increase the chunk_size and max_concurrency to optimize processing time
def analyze_batch_langchain(df: pd.DataFrame, n_tweets=None, chunk_size=4):
    if n_tweets:
        df = df.head(n_tweets).copy()
    
    inputs = [
        {"tweet": row['cleaned_tweet'], "company": row['company'], "author": row['author']} 
        for _, row in df.iterrows()
    ]
    
    all_results = []
    
    # Process in smaller chunks to avoid hitting the "Per Minute" (RPM) limit
    for i in range(0, len(inputs), chunk_size):
        chunk = inputs[i : i + chunk_size]
        
        # Lowered max_concurrency to 2 to be very safe
        chunk_results = sentiment_chain.batch(chunk, config={"max_concurrency": 1})
        all_results.extend(chunk_results)
        
        # Optional: Add a small sleep between batches if the error persists
        # time.sleep(2) 
    
    enriched_df = pd.DataFrame(all_results)
    
    # Ensure indices match for joining
    enriched_df.index = df.index
    final_df = df.join(enriched_df, rsuffix='_result')
    
    # final_df.to_json('TFM/gemini_tweets_final.json', orient='records', lines=True, force_ascii=False)
    final_df[['tweet','sentiment','subject','urgency','summary']].to_json('TFM/sentiment_chain_tweets_test.json', orient='records', lines=True, force_ascii=False)
    return final_df

# 6. Execute the batch functions - UNCOMMENT THIS SECTION TO TEST THE FUNCTIONS
# result = analyze_batch_langchain(df_test)
