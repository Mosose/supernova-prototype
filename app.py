import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from textblob import TextBlob

# Set up the UI
st.set_page_config(page_title="Digital Agorá", page_icon="🏛️")
st.title("🏛️ The Supernova Synthesizer")
st.subheader("Transforming 'Crowd Noise' into 'Democratic Signal'")

# 1. The Input Section
st.write("### 📥 Step 1: Simulated Public Input (The Crowd)")
raw_data = [
    "Parking is a nightmare, build the lot.", 
    "A park would be great for the children.",
    "We need more trees to combat the urban heat.",
    "Subterranean parking with a park on top is the future!",
    "BUILD PARKING NOW!", # Bot simulation
    "BUILD PARKING NOW!"  # Bot simulation
]
df = pd.DataFrame(raw_data, columns=["Raw Opinions"])
st.dataframe(df, use_container_width=True)

if st.button("Synthesize Opinions"):
    with st.spinner("Filtering bots and analyzing semantics..."):
        
        # 2. Integrity Filter (Bot Removal)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        st.success(f"🛡️ Integrity Filter: Removed {len(df) - len(df_clean)} duplicate/bot entries.")
        
        # 3. AI Processing (Embeddings & Sentiment)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df_clean["Raw Opinions"].tolist())
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(embeddings)
        df_clean['Cluster'] = kmeans.labels_
        
       # A smarter "Urgency" proxy
def calculate_urgency(text):
    # Base subjectivity score
    score = TextBlob(text).sentiment.subjectivity * 10
    
    # "Urgency Booster" for high-emotion keywords
    urgency_words = ['nightmare', 'now', 'urgent', 'must', 'immediately', '!', 'angry']
    for word in urgency_words:
        if word in text.lower():
            score += 3.0 # Boost the score for each urgent word
            
    return min(float(score), 10.0) # Cap it at 10.0

df_clean['Urgency (1-10)'] = df_clean["Raw Opinions"].apply(calculate_urgency)
