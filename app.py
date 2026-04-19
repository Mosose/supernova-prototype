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
uploaded_file = st.file_uploader("Upload a CSV of community comments", type=["csv"])

# Dynamic Cluster Slider (Lets the user choose how many issues to look for)
num_clusters = st.slider("How many 'Signal Groups' should the AI find?", min_value=2, max_value=10, value=3)

if uploaded_file is not None:
    # If they upload real data, use it!
    df = pd.read_csv(uploaded_file)
    # Assuming the CSV has a column called 'text' or we just take the first column
    column_name = df.columns[0] 
    df = df.rename(columns={column_name: "Raw Opinions"})
    st.success(f"Loaded {len(df)} real complaints!")
    st.dataframe(df.head(), use_container_width=True)
else:
    # If they don't upload a file, use the demo data
    st.info("No file uploaded. Using demo dataset:")
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
        
        # TITANIUM-PLATED CLEANER
        # 1. Force convert everything to standard python strings and strip whitespace/hidden characters
        df_clean["Raw Opinions"] = [str(x).strip() for x in df_clean["Raw Opinions"]]
        
        # 2. Drop anything that is empty, "nan", "none", or too short to be a real sentence
        df_clean = df_clean[df_clean["Raw Opinions"].str.lower() != "nan"]
        df_clean = df_clean[df_clean["Raw Opinions"].str.lower() != "none"]
        df_clean = df_clean[df_clean["Raw Opinions"].str.len() > 5] # Must be at least 5 characters
        df_clean = df_clean.reset_index(drop=True)
        
        # 3. Failsafe: Did the CSV formatting delete everything?
        if len(df_clean) == 0:
            st.error("🚨 Critical Error: The uploaded file contained no valid text after cleaning. The CSV is likely corrupted by a text editor. Please try re-saving it in Excel or Apple Numbers.")
            st.stop() # Stops the app before it hits the AI to prevent crashing
            
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
        
        # Only show the demo "Supernova Insight" if using the demo data
        if uploaded_file is None:
            st.info("🌟 **Supernova Insight:** Notice how the AI separated the 'Subterranean parking' idea as its own distinct cluster. It didn't erase the minority compromise.")
