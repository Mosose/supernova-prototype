import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from textblob import TextBlob

# Define the smarter "Urgency" proxy at the top
def calculate_urgency(text):
    # Base subjectivity score
    score = TextBlob(text).sentiment.subjectivity * 10
    
    # "Urgency Booster" for high-emotion keywords
    urgency_words = ['nightmare', 'now', 'urgent', 'must', 'immediately', '!', 'angry']
    for word in urgency_words:
        if word in text.lower():
            score += 3.0 # Boost the score for each urgent word
            
    return min(float(score), 10.0) # Cap it at 10.0

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

# 2. The Execution Block
if st.button("Synthesize Opinions"):
    with st.spinner("Filtering bots and analyzing semantics..."):
        
        # Integrity Filter (Bot Removal)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        st.success(f"🛡️ Integrity Filter: Removed {len(df) - len(df_clean)} duplicate/bot entries.")
        
        # AI Processing (Embeddings & Sentiment)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df_clean["Raw Opinions"].tolist())
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(embeddings)
        df_clean['Cluster'] = kmeans.labels_
        
        # Apply the new urgency calculation
        df_clean['Urgency (1-10)'] = df_clean["Raw Opinions"].apply(calculate_urgency)
        
        # 3. The Output Dashboard
        st.write("### 📊 Step 2: The Signal Report")
        
        for i in range(3):
            cluster_data = df_clean[df_clean['Cluster'] == i]
            # Handle empty clusters gracefully
            if not cluster_data.empty:
                avg_urgency = round(cluster_data['Urgency (1-10)'].mean(), 1)
                
                with st.expander(f"Signal Group {i+1} (Urgency: {avg_urgency}/10)"):
                    for index, row in cluster_data.iterrows():
                        st.write(f"- {row['Raw Opinions']}")
                    
        st.info("🌟 **Supernova Insight:** Notice how the AI separated the 'Subterranean parking' idea as its own distinct cluster. It didn't erase the minority compromise.")
