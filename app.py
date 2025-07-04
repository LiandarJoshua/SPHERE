import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Hybrid Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for models
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.als_model = None
    st.session_state.bpr_model = None
    st.session_state.confidence_matrix = None
    st.session_state.user_mapper = None
    st.session_state.item_mapper = None
    st.session_state.item_inv_mapper = None
    st.session_state.sparse_matrix = None

class RecommendationSystem:
    def __init__(self):
        self.als_model = None
        self.bpr_model = None
        self.confidence_matrix = None
        self.user_mapper = {}
        self.item_mapper = {}
        self.item_inv_mapper = {}
        self.sparse_matrix = None
        self.df = None
        
    def load_data(self, uploaded_file):
        """Load and preprocess data"""
        try:
            if uploaded_file.name.endswith('.csv'):
                self.df = pd.read_csv(uploaded_file)
            else:
                self.df = pd.read_excel(uploaded_file)
            
            # Assume columns: user_id, item_id, rating/interaction
            # Adjust column names as needed
            if 'user_id' not in self.df.columns:
                self.df.rename(columns={self.df.columns[0]: 'user_id'}, inplace=True)
            if 'item_id' not in self.df.columns:
                self.df.rename(columns={self.df.columns[1]: 'item_id'}, inplace=True)
            if 'rating' not in self.df.columns:
                self.df.rename(columns={self.df.columns[2]: 'rating'}, inplace=True)
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def create_mappings(self):
        """Create user and item mappings"""
        unique_users = self.df['user_id'].unique()
        unique_items = self.df['item_id'].unique()
        
        self.user_mapper = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapper = {item: idx for idx, item in enumerate(unique_items)}
        self.item_inv_mapper = {idx: item for item, idx in self.item_mapper.items()}
        
        # Create sparse matrix
        rows = [self.user_mapper[user] for user in self.df['user_id']]
        cols = [self.item_mapper[item] for item in self.df['item_id']]
        data = self.df['rating'].values
        
        self.sparse_matrix = sp.csr_matrix(
            (data, (rows, cols)), 
            shape=(len(unique_users), len(unique_items))
        )
        
    def train_models(self):
        """Train ALS and BPR models with popularity weighting"""
        # Step 1: Get item popularity
        item_popularity = np.array(self.sparse_matrix.sum(axis=0)).flatten()
        
        # Step 2: Compute inverse popularity weights
        inv_weights = 1 / (item_popularity + 1e-6)
        
        # Step 3: Apply weights to interaction values
        confidence_matrix = self.sparse_matrix.copy()
        confidence_matrix = confidence_matrix.tocoo()
        
        weighted_data = confidence_matrix.data * inv_weights[confidence_matrix.col]
        self.confidence_matrix = sp.coo_matrix(
            (weighted_data, (confidence_matrix.row, confidence_matrix.col)),
            shape=confidence_matrix.shape
        ).tocsr()
        
        # Step 4: Train models
        self.als_model = AlternatingLeastSquares(
            factors=50,
            regularization=0.1,
            iterations=15,
            use_gpu=False
        )
        self.als_model.fit(self.confidence_matrix)
        
        self.bpr_model = BayesianPersonalizedRanking(
            factors=50,
            regularization=0.01,
            learning_rate=0.01,
            iterations=20
        )
        self.bpr_model.fit(self.confidence_matrix)
        
    def get_cbf_recommendations(self, user_id, N=5):
        """Placeholder for content-based filtering"""
        # This would be replaced with your actual CBF implementation
        user_items = self.df[self.df['user_id'] == user_id]['item_id'].values
        if len(user_items) == 0:
            return []
        
        # Simple popularity-based fallback
        item_counts = self.df['item_id'].value_counts()
        popular_items = item_counts.head(N*2).index.tolist()
        
        # Remove items user already interacted with
        recommendations = [item for item in popular_items if item not in user_items]
        return recommendations[:N]
    
    def hybrid_recommend(self, user_id, alpha=0.4, beta=0.3, N=5):
        """Hybrid recommendation combining CBF, ALS, and BPR"""
        try:
            if user_id not in self.user_mapper:
                return []
                
            user_idx = self.user_mapper[user_id]
            
            cbf_items = self.get_cbf_recommendations(user_id, N)
            
            als_recs = self.als_model.recommend(
                user_idx, 
                self.sparse_matrix[user_idx], 
                N=N
            )
            als_items = [self.item_inv_mapper[int(i)] for i in als_recs[0]]
            als_scores = als_recs[1]
            
            bpr_recs = self.bpr_model.recommend(
                user_idx, 
                self.sparse_matrix[user_idx], 
                N=N
            )
            bpr_items = [self.item_inv_mapper[int(i)] for i in bpr_recs[0]]
            bpr_scores = bpr_recs[1]
            
            scores = {}
            # CBF scores (rank-based)
            for i, item in enumerate(cbf_items):
                scores[item] = scores.get(item, 0) + alpha * (N - i)
            
            # ALS scores
            for item, score in zip(als_items, als_scores):
                scores[item] = scores.get(item, 0) + beta * score
            
            # BPR scores
            for item, score in zip(bpr_items, bpr_scores):
                scores[item] = scores.get(item, 0) + (1 - alpha - beta) * score
            
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [item for item, _ in ranked[:N]]
            
        except Exception as e:
            st.error(f"Error in hybrid_recommend for user {user_id}: {e}")
            return []

# Initialize recommendation system
@st.cache_resource
def get_recommendation_system():
    return RecommendationSystem()

rec_system = get_recommendation_system()

# Main app
def main():
    st.markdown('<h1 class="main-header">🎯 Hybrid Recommendation System Demo</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Data upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload interaction data",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file with columns: user_id, item_id, rating"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("Load Data"):
            with st.spinner("Loading data..."):
                if rec_system.load_data(uploaded_file):
                    st.success("Data loaded successfully!")
                    
                    # Display data info
                    st.subheader("📊 Dataset Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Interactions", len(rec_system.df))
                    with col2:
                        st.metric("Unique Users", rec_system.df['user_id'].nunique())
                    with col3:
                        st.metric("Unique Items", rec_system.df['item_id'].nunique())
                    with col4:
                        st.metric("Avg Rating", f"{rec_system.df['rating'].mean():.2f}")
                    
                    # Show sample data
                    st.subheader("Sample Data")
                    st.dataframe(rec_system.df.head())
                    
                    # Data visualization
                    st.subheader("📈 Data Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Rating distribution
                        fig_rating = px.histogram(
                            rec_system.df, 
                            x='rating', 
                            title='Rating Distribution',
                            nbins=20
                        )
                        st.plotly_chart(fig_rating, use_container_width=True)
                    
                    with col2:
                        # Top items
                        top_items = rec_system.df['item_id'].value_counts().head(10)
                        fig_items = px.bar(
                            x=top_items.index, 
                            y=top_items.values,
                            title='Top 10 Most Popular Items'
                        )
                        fig_items.update_layout(xaxis_title='Item ID', yaxis_title='Interactions')
                        st.plotly_chart(fig_items, use_container_width=True)
    
    # Model training section
    if uploaded_file is not None and rec_system.df is not None:
        st.sidebar.header("Model Training")
        
        if st.sidebar.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                rec_system.create_mappings()
                rec_system.train_models()
                
                # Store in session state
                st.session_state.models_trained = True
                st.session_state.als_model = rec_system.als_model
                st.session_state.bpr_model = rec_system.bpr_model
                st.session_state.confidence_matrix = rec_system.confidence_matrix
                st.session_state.user_mapper = rec_system.user_mapper
                st.session_state.item_mapper = rec_system.item_mapper
                st.session_state.item_inv_mapper = rec_system.item_inv_mapper
                st.session_state.sparse_matrix = rec_system.sparse_matrix
                
                st.success("Models trained successfully!")
                
                # Model info
                st.subheader("🤖 Model Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="metric-container">
                    <h4>ALS Model</h4>
                    <p>• Factors: 50</p>
                    <p>• Regularization: 0.1</p>
                    <p>• Iterations: 15</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-container">
                    <h4>BPR Model</h4>
                    <p>• Factors: 50</p>
                    <p>• Regularization: 0.01</p>
                    <p>• Learning Rate: 0.01</p>
                    <p>• Iterations: 20</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Recommendation section
    if st.session_state.models_trained:
        st.subheader("🎯 Get Recommendations")
        
        # User selection
        available_users = list(rec_system.user_mapper.keys())
        selected_user = st.selectbox("Select a user:", available_users)
        
        # Hybrid weights
        st.subheader("⚙️ Hybrid Model Weights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.1)
        with col2:
            beta = st.slider("ALS Weight", 0.0, 1.0, 0.3, 0.1)
        with col3:
            gamma = 1 - alpha - beta
            st.metric("BPR Weight", f"{gamma:.1f}")
        
        # Number of recommendations
        N = st.slider("Number of recommendations", 1, 20, 5)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = rec_system.hybrid_recommend(
                    selected_user, alpha=alpha, beta=beta, N=N
                )
                
                if recommendations:
                    st.subheader(f"🎉 Top {N} Recommendations for User {selected_user}")
                    
                    # Display recommendations
                    for i, item in enumerate(recommendations, 1):
                        st.markdown(f"""
                        <div class="recommendation-box">
                        <h4>#{i} - Item: {item}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show user's interaction history
                    st.subheader("📋 User's Interaction History")
                    user_history = rec_system.df[rec_system.df['user_id'] == selected_user]
                    if not user_history.empty:
                        st.dataframe(user_history.sort_values('rating', ascending=False))
                    else:
                        st.info("No interaction history found for this user.")
                else:
                    st.warning("No recommendations generated. Please check the user ID and model training.")
    
    # Model comparison section
    if st.session_state.models_trained:
        st.subheader("📊 Model Comparison")
        
        if st.button("Compare Models"):
            sample_users = list(rec_system.user_mapper.keys())[:5]  # Sample 5 users
            
            comparison_data = []
            for user in sample_users:
                # Get recommendations from each model
                cbf_recs = rec_system.get_cbf_recommendations(user, 5)
                
                user_idx = rec_system.user_mapper[user]
                als_recs = rec_system.als_model.recommend(user_idx, rec_system.sparse_matrix[user_idx], N=5)
                als_items = [rec_system.item_inv_mapper[int(i)] for i in als_recs[0]]
                
                bpr_recs = rec_system.bpr_model.recommend(user_idx, rec_system.sparse_matrix[user_idx], N=5)
                bpr_items = [rec_system.item_inv_mapper[int(i)] for i in bpr_recs[0]]
                
                hybrid_recs = rec_system.hybrid_recommend(user, N=5)
                
                comparison_data.append({
                    'User': user,
                    'CBF': ', '.join(map(str, cbf_recs[:3])),
                    'ALS': ', '.join(map(str, als_items[:3])),
                    'BPR': ', '.join(map(str, bpr_items[:3])),
                    'Hybrid': ', '.join(map(str, hybrid_recs[:3]))
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with Streamlit** | Hybrid Recommendation System Demo")

if __name__ == "__main__":
    main()