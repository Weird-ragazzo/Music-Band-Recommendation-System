import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# --------- Page Configuration --------- #
st.set_page_config(
    page_title="Band Recommendation System",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------- Custom CSS for Modern Styling --------- #
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .sidebar .stSelectbox > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-message {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------- Load datasets --------- #
@st.cache_data
def load_data():
    try:
        # Load preprocessed dataset (with encoded features + Band column)
        pre_df = pd.read_csv('preprocessed_band_dataset.csv')
        
        # Load original dataset with Band details
        orig_df = pd.read_csv('alternative_metal_bands.csv')
        
        # Merge to keep all details along with encoded features
        df = pd.merge(orig_df, pre_df, on='Band')
        
        return df, orig_df
    except FileNotFoundError as e:
        st.error(f"Dataset not found: {e}")
        return None, None

# --------- Function to build user input vector --------- #
def build_user_vector(user_active, user_origin, user_genres, origin_cols, genre_cols):
    active_val = 1 if user_active == 'Yes' else 0

    origin_vector = [0] * len(origin_cols)
    user_origin_col = 'Origin_' + user_origin
    if user_origin_col in origin_cols:
        idx = origin_cols.index(user_origin_col)
        origin_vector[idx] = 1

    genres_list = [g.strip() for g in user_genres.split(',')]
    genre_vector = [0] * len(genre_cols)
    for i, g in enumerate(genre_cols):
        if g in genres_list:
            genre_vector[i] = 1

    return [active_val] + origin_vector + genre_vector

# --------- Main Application --------- #
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎸 Band Recommendation System</h1>
        <p>Discover new bands based on your musical preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df, orig_df = load_data()
    
    if df is None or orig_df is None:
        st.error("Unable to load datasets. Please check if the files exist.")
        return
    
    # Sidebar for user inputs
    with st.sidebar:
        st.markdown("### 🎵 Your Preferences")
        
        # User Inputs
        user_band = st.text_input(
            "Your favourite band name",
            placeholder="Enter band name...",
            help="This will be excluded from recommendations"
        )
        
        user_active = st.selectbox(
            "Is the band active?",
            ["Yes", "No"],
            help="Select if the band is currently active"
        )
        
        user_origin = st.selectbox(
            "Band origin",
            sorted(orig_df['Origin'].unique()),
            help="Select the country/region of origin"
        )
        
        user_genres = st.text_input(
            "Genres (comma-separated)",
            placeholder="e.g., Metal, Rock, Alternative",
            help="Enter genres separated by commas"
        )
        
        # Statistics
        st.markdown("### 📊 Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Bands", len(df))
        with col2:
            st.metric("Countries", len(orig_df['Origin'].unique()))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Find Recommendations", type="primary"):
            if user_band and user_genres:
                with st.spinner("Analyzing musical preferences..."):
                    # Prepare features dataframe for similarity
                    features_df = df.drop(columns=['Band', 'Active_x', 'Origin', 'Genres'])
                    
                    # Identify origin and genre columns dynamically
                    origin_cols = [col for col in features_df.columns if col.startswith('Origin_')]
                    genre_cols = [col for col in features_df.columns if col not in ['Active_y'] + origin_cols]
                    
                    # Build user vector
                    user_vector = build_user_vector(user_active, user_origin, user_genres, origin_cols, genre_cols)
                    user_df = pd.DataFrame([user_vector], columns=['Active_y'] + origin_cols + genre_cols)
                    
                    # Calculate similarity
                    sim = cosine_similarity(user_df, features_df)[0]
                    df['Similarity'] = sim
                    
                    # Sort by similarity
                    results = df.sort_values(by='Similarity', ascending=False)
                    
                    # Exclude input band
                    results = results[results['Band'] != user_band]
                    
                    # Get top recommendations
                    top_recommendations = results.head(10)
                    
                    # Display results
                    st.markdown("### 🎶 Top Recommendations")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📋 Detailed View", "📊 Similarity Chart", "🗺️ Origin Map"])
                    
                    with tab1:
                        # Display detailed recommendations
                        display_cols = ['Band', 'Active_x', 'Origin', 'Genres', 'Similarity']
                        recommendations_df = top_recommendations[display_cols].head(5).reset_index(drop=True)
                        
                        # Format similarity as percentage
                        recommendations_df['Similarity'] = recommendations_df['Similarity'].apply(lambda x: f"{x:.1%}")
                        
                        # Rename columns for better display
                        recommendations_df.columns = ['Band', 'Status', 'Origin', 'Genres', 'Match %']
                        
                        st.dataframe(
                            recommendations_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Show individual recommendation cards
                        st.markdown("### 🎯 Recommendation Details")
                        for idx, row in recommendations_df.head(3).iterrows():
                            st.markdown(f"""
                            <div class="recommendation-card">
                                <h4>🎸 {row['Band']}</h4>
                                <p><strong>Origin:</strong> {row['Origin']} | <strong>Status:</strong> {row['Status']} | <strong>Match:</strong> {row['Match %']}</p>
                                <p><strong>Genres:</strong> {row['Genres']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab2:
                        # Similarity chart
                        chart_data = top_recommendations[['Band', 'Similarity']].head(8)
                        
                        fig = px.bar(
                            chart_data,
                            x='Similarity',
                            y='Band',
                            orientation='h',
                            title='Similarity Scores',
                            color='Similarity',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(
                            height=400,
                            showlegend=False,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Origin distribution
                        origin_counts = top_recommendations['Origin'].value_counts().head(10)
                        
                        fig = px.pie(
                            values=origin_counts.values,
                            names=origin_counts.index,
                            title='Recommended Bands by Origin'
                        )
                        
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(height=400)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Success message
                    st.success(f"✅ Found {len(results)} similar bands based on your preferences!")
                    
            else:
                st.warning("⚠️ Please enter both band name and genres to get recommendations.")
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **Getting Better Recommendations:**
        
        🎯 **Be Specific**: Use detailed genre names
        
        🌍 **Origin Matters**: Geographic location influences musical style
        
        🎸 **Multiple Genres**: Separate genres with commas
        
        ⭐ **Explore**: Try different combinations!
        """)
        
        # Sample genres
        st.markdown("### 🎵 Popular Genres")
        sample_genres = ["Metal", "Alternative Metal", "Heavy Metal", "Progressive Metal", "Death Metal", "Black Metal"]
        for genre in sample_genres:
            if st.button(f"#{genre}", key=f"genre_{genre}"):
                st.session_state.user_genres = genre

if __name__ == "__main__":
    main()