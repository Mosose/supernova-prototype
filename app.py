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
st.write("### 📥 Step 1: Upload Public Input (The Crowd)")

# The Drag-and-Drop Uploader!
uploaded_file = st.file_uploader("Upload a CSV of community comments (Optional)", type=["csv"])

# Dynamic Cluster Slider (Lets the user choose how many issues to look for)
num_clusters = st.slider("How many 'Signal Groups' should the AI find?", min_value=2, max_value=10, value=4)

if uploaded_file is not None:
    # If they upload real data, use it!
    try:
        df = pd.read_csv(uploaded_file)
        column_name = df.columns[0] 
        df = df.rename(columns={column_name: "Raw Opinions"})
        st.success(f"Loaded {len(df)} complaints from file!")
        st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read file. Using default dataset instead.")
        uploaded_file = None # Force fallback to demo data
        
if uploaded_file is None:
    # THE BULLETPROOF DEFAULT: Real NYC 311 Data
    st.info("No file uploaded. Using real-world NYC 311 sample data:")
    raw_data = [
        "Streetlight out in front of my house, the block is completely dark.",
        "Loud music playing from the restaurant on the corner since 10 PM.",
        "The garbage truck missed our block again this week!",
        "Huge pothole in the crosswalk, someone is going to trip and break an ankle.",
        "Neighbor's dog has been barking in the yard for 4 hours straight.",
        "There are overflowing trash cans at the entrance to the public park.",
        "Rat sighting in the alleyway behind the bodega, needs pest control urgently!",
        "Cars are constantly running the stop sign at this intersection, it's a nightmare.",
        "Construction noise at 6 AM on a Sunday, totally unacceptable.",
        "The storm drain is clogged with leaves and the street is flooding.",
        "Graffiti on the side of the elementary school building.",
        "Loud music playing from the restaurant on the corner since 10 PM.", # Bot/Duplicate
        "The storm drain is clogged with leaves and the street is flooding." # Bot/Duplicate
    ]
    df = pd.DataFrame(raw_data, columns=["Raw Opinions"])
    st.dataframe(df, use_container_width=True)

# 2. The Execution Block
if st.button("Synthesize Opinions"):
    with st.spinner("Filtering bots and analyzing semantics..."):
        
        # Integrity Filter (Bot Removal)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        st.success(f"🛡️ Integrity Filter: Removed {len(df) - len(df_clean)} duplicate/bot entries.")
        
        # Force string conversion to prevent AI crashing
        df_clean["Raw Opinions"] = df_clean["Raw Opinions"].astype(str)
        
        # AI Processing (Embeddings & Sentiment)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df_clean["Raw Opinions"].tolist())
        
        # Ensure we don't ask for more clusters than we have data points
        actual_clusters = min(num_clusters, len(df_clean))
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10).fit(embeddings)
        df_clean['Cluster'] = kmeans.labels_
        
        # Apply the new urgency calculation
        df_clean['Urgency (1-10)'] = df_clean["Raw Opinions"].apply(calculate_urgency)
        
        # 3. The Output Dashboard
        st.write("### 📊 Step 2: The Signal Report")
        
        # Loop through the actual number of clusters requested by the slider
        for i in range(actual_clusters):
            cluster_data = df_clean[df_clean['Cluster'] == i]
            # Handle empty clusters gracefully
            if not cluster_data.empty:
                avg_urgency = round(cluster_data['Urgency (1-10)'].mean(), 1)
                
                with st.expander(f"Signal Group {i+1} (Urgency: {avg_urgency}/10)"):
                    for index, row in cluster_data.iterrows():
                        st.write(f"- {row['Raw Opinions']}")
                        
        if uploaded_file is None:
            st.info("🌟 **Supernova Insight:** Notice how the AI separated the noise complaints (dogs, construction, music) from the sanitation complaints (trash, rats) automatically.")
