import streamlit as st
import pandas as pd
import PyPDF2
import numpy as np
import faiss
import folium
from streamlit_folium import st_folium
from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory
import io
import re
from groq import Groq
from typing import List, Dict, Optional
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
DetectorFactory.seed = 0

# Page configuration
st.set_page_config(
    page_title="Campus Information Assistant",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'location_data' not in st.session_state:
    st.session_state.location_data = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'multilingual_model' not in st.session_state:
    st.session_state.multilingual_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load models with better error handling
@st.cache_resource
def load_models():
    """Load sentence transformer models with fallback options"""
    try:
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            # Try to load primary models
            try:
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                st.success("✅ English model loaded successfully")
            except Exception as e:
                st.error(f"Failed to load English model: {str(e)}")
                model = None
            
            try:
                multilingual_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')
                st.success("✅ Multilingual model loaded successfully")
            except Exception as e:
                st.warning(f"Failed to load multilingual model: {str(e)} - Using English model for all queries")
                multilingual_model = model
            
            return model, multilingual_model
    except Exception as e:
        st.error(f"Critical error loading models: {str(e)}")
        return None, None

def get_groq_api_key():
    """Get Groq API key from secrets or sidebar"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return st.sidebar.text_input("Enter Groq API Key", type="password", help="Get your API key from https://groq.com")

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file with better error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                st.warning(f"Could not extract text from page {i+1}: {str(e)}")
                continue
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return ""

def process_text_to_sentences(text: str) -> List[str]:
    """Split text into meaningful chunks for RAG"""
    if not text.strip():
        return []
    
    # Split by sentences and paragraphs
    sentences = re.split(r'[.!?]+|\n\n+', text)
    
    # Clean and filter sentences
    processed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Keep sentences that are meaningful (20-500 characters)
        if 20 <= len(sentence) <= 500:
            processed_sentences.append(sentence)
    
    return processed_sentences

def process_csv_excel_file(file) -> tuple:
    """Process CSV/Excel files and extract both text and location data"""
    try:
        # Read the file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Extract location data
        locations = extract_location_data(df)
        
        # Convert DataFrame to text for RAG
        text_data = df.to_string(index=False)
        sentences = process_text_to_sentences(text_data)
        
        return sentences, locations, df
    except Exception as e:
        st.error(f"Error processing file {file.name}: {str(e)}")
        return [], [], pd.DataFrame()

def extract_location_data(df: pd.DataFrame) -> List[Dict]:
    """Extract location information from DataFrame"""
    locations = []
    
    if df.empty:
        return locations
    
    # Convert column names to lowercase for matching
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    # Try to find location-related columns
    name_col = None
    lat_col = None
    lon_col = None
    desc_col = None
    
    # Look for name/location columns
    for name_variant in ['name', 'location', 'place', 'building', 'facility', 'title']:
        if name_variant in df_cols_lower:
            name_col = df_cols_lower[name_variant]
            break
    
    # Look for latitude columns
    for lat_variant in ['latitude', 'lat', 'y']:
        if lat_variant in df_cols_lower:
            lat_col = df_cols_lower[lat_variant]
            break
    
    # Look for longitude columns
    for lon_variant in ['longitude', 'lon', 'lng', 'long', 'x']:
        if lon_variant in df_cols_lower:
            lon_col = df_cols_lower[lon_variant]
            break
    
    # Look for description columns
    for desc_variant in ['description', 'desc', 'info', 'details', 'about']:
        if desc_variant in df_cols_lower:
            desc_col = df_cols_lower[desc_variant]
            break
    
    # Extract locations if we have the required columns
    if name_col and lat_col and lon_col:
        for _, row in df.iterrows():
            try:
                location = {
                    'name': str(row[name_col]).strip(),
                    'latitude': float(row[lat_col]),
                    'longitude': float(row[lon_col]),
                    'description': str(row[desc_col]).strip() if desc_col else ""
                }
                
                # Validate coordinates
                if -90 <= location['latitude'] <= 90 and -180 <= location['longitude'] <= 180:
                    locations.append(location)
            except (ValueError, TypeError):
                continue
    
    return locations

def create_faiss_index(texts: List[str], model) -> Optional[faiss.IndexFlatIP]:
    """Create FAISS index for similarity search"""
    if not texts or not model:
        return None
    
    try:
        with st.spinner("Creating search index..."):
            # Create embeddings
            embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            
            return index
    except Exception as e:
        st.error(f"Error creating search index: {str(e)}")
        return None

def search_knowledge_base(query: str, index, knowledge_base: List[str], model, k: int = 3) -> List[str]:
    """Search knowledge base using FAISS"""
    if not index or not knowledge_base or not model or not query.strip():
        return []
    
    try:
        # Create query embedding
        query_embedding = model.encode([query], convert_to_tensor=False, show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding, min(k, len(knowledge_base)))
        
        # Return results
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(knowledge_base):
                results.append(knowledge_base[idx])
        
        return results
    except Exception as e:
        st.error(f"Error searching knowledge base: {str(e)}")
        return []

def find_location_in_query(query: str, locations: List[Dict]) -> Optional[Dict]:
    """Find if query mentions any known location"""
    if not query or not locations:
        return None
    
    query_lower = query.lower()
    
    # Look for exact matches first
    for location in locations:
        if location['name'].lower() in query_lower:
            return location
    
    # Look for partial matches
    for location in locations:
        name_words = location['name'].lower().split()
        for word in name_words:
            if len(word) > 3 and word in query_lower:
                return location
    
    return None

def generate_response_with_groq(query: str, context: List[str], groq_api_key: str) -> str:
    """Generate response using Groq API"""
    if not groq_api_key:
        return "Please provide your Groq API key to enable AI responses. You can get one free at https://groq.com"
    
    try:
        client = Groq(api_key=groq_api_key)
        
        # Prepare context
        context_text = "\n".join(context[:3]) if context else "No specific context available."
        
        prompt = f"""You are a helpful campus assistant. Answer the user's question based on the provided context. If the context doesn't contain relevant information, provide a helpful general response and mention that specific campus information might not be available.

Context from campus documents:
{context_text}

User Question: {query}

Provide a helpful, friendly response:"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't generate a response right now. Error: {str(e)}"

def admin_page():
    """Admin page for file upload and processing"""
    st.title("🔧 Admin Dashboard")
    st.markdown("Upload campus documents and location data to build the knowledge base.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Files (PDF, CSV, Excel)",
        type=['pdf', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload campus documents for Q&A and CSV/Excel files with location data"
    )
    
    if uploaded_files:
        st.subheader("Processing Files")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_texts = []
        all_locations = []
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            try:
                if file.type == "application/pdf":
                    # Process PDF
                    text = extract_text_from_pdf(file)
                    if text:
                        sentences = process_text_to_sentences(text)
                        all_texts.extend(sentences)
                        st.success(f"✅ {file.name}: Extracted {len(sentences)} text segments")
                    else:
                        st.warning(f"⚠️ {file.name}: No text could be extracted")
                
                elif file.type in ["text/csv", "application/vnd.ms-excel", 
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                    # Process CSV/Excel
                    sentences, locations, df = process_csv_excel_file(file)
                    
                    if not df.empty:
                        st.success(f"✅ {file.name}: Loaded {len(df)} rows")
                        with st.expander(f"Preview of {file.name}"):
                            st.dataframe(df.head())
                    
                    if sentences:
                        all_texts.extend(sentences)
                        st.info(f"📝 {file.name}: Added {len(sentences)} text segments for search")
                    
                    if locations:
                        all_locations.extend(locations)
                        st.info(f"📍 {file.name}: Found {len(locations)} locations")
            
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Update session state
        if all_texts:
            st.session_state.knowledge_base.extend(all_texts)
            st.success(f"📚 Added {len(all_texts)} text segments to knowledge base")
        
        if all_locations:
            st.session_state.location_data.extend(all_locations)
            st.success(f"🗺️ Added {len(all_locations)} locations to database")
        
        # Create/update FAISS index
        if st.session_state.knowledge_base and st.session_state.model:
            status_text.text("Building search index...")
            st.session_state.faiss_index = create_faiss_index(
                st.session_state.knowledge_base, 
                st.session_state.model
            )
            if st.session_state.faiss_index:
                st.success("🔍 Search index created successfully!")
        
        status_text.text("✅ Processing complete!")
    
    # Display current statistics
    st.subheader("📊 Knowledge Base Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Text Segments", len(st.session_state.knowledge_base))
        
    with col2:
        st.metric("Locations", len(st.session_state.location_data))
    
    with col3:
        index_status = "Ready" if st.session_state.faiss_index else "Not Created"
        st.metric("Search Index", index_status)
    
    # Display locations if available
    if st.session_state.location_data:
        st.subheader("📍 Campus Locations")
        locations_df = pd.DataFrame(st.session_state.location_data)
        st.dataframe(locations_df, use_container_width=True)
        
        # Show locations on map
        if st.checkbox("Show all locations on map"):
            center_lat = locations_df['latitude'].mean()
            center_lon = locations_df['longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
            
            for _, location in locations_df.iterrows():
                folium.Marker(
                    [location['latitude'], location['longitude']],
                    popup=f"<b>{location['name']}</b><br>{location['description']}",
                    tooltip=location['name']
                ).add_to(m)
            
            st_folium(m, height=400, width=700)
    
    # Clear data options
    st.subheader("🗑️ Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Text Data"):
            st.session_state.knowledge_base = []
            st.session_state.faiss_index = None
            st.success("Text data cleared!")
            st.rerun()
    
    with col2:
        if st.button("Clear Location Data"):
            st.session_state.location_data = []
            st.success("Location data cleared!")
            st.rerun()

def user_page():
    """User page with chat interface"""
    st.title("💬 Campus Assistant")
    st.markdown("Ask me anything about the campus! I can help with information and directions.")
    
    # Display chat history
    for i, (user_msg, bot_msg, location_info) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(user_msg)
        
        with st.chat_message("assistant"):
            st.write(bot_msg)
            
            # Show location and map if available
            if location_info:
                st.subheader(f"📍 {location_info['name']}")
                if location_info['description']:
                    st.write(f"*{location_info['description']}*")
                
                # Create map
                m = folium.Map(
                    location=[location_info['latitude'], location_info['longitude']], 
                    zoom_start=17
                )
                folium.Marker(
                    [location_info['latitude'], location_info['longitude']],
                    popup=f"<b>{location_info['name']}</b>",
                    tooltip=location_info['name'],
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
                
                st_folium(m, height=300, width=700)
    
    # Chat input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Detect language
        try:
            detected_lang = detect(user_input)
            is_english = detected_lang == 'en'
        except:
            is_english = True
        
        # Choose model
        search_model = st.session_state.model
        if not is_english and st.session_state.multilingual_model:
            search_model = st.session_state.multilingual_model
        
        # Search knowledge base
        relevant_context = []
        if st.session_state.faiss_index and search_model:
            relevant_context = search_knowledge_base(
                user_input, 
                st.session_state.faiss_index, 
                st.session_state.knowledge_base, 
                search_model,
                k=3
            )
        
        # Check for location mentions
        location_info = find_location_in_query(user_input, st.session_state.location_data)
        
        # Generate response
        groq_api_key = get_groq_api_key()
        response = generate_response_with_groq(user_input, relevant_context, groq_api_key)
        
        # Add location info to response if found
        if location_info and not any(location_info['name'].lower() in response.lower() for _ in [0]):
            response += f"\n\n📍 I found information about **{location_info['name']}** - check the map below!"
        
        # Add to chat history
        st.session_state.chat_history.append((user_input, response, location_info))
        
        st.rerun()

def main():
    """Main application function"""
    # Load models on startup
    if st.session_state.model is None:
        st.session_state.model, st.session_state.multilingual_model = load_models()
    
    # Sidebar
    st.sidebar.title("🏫 Campus Assistant")
    st.sidebar.markdown("---")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page", 
        ["👤 User Chat", "🔧 Admin Panel"],
        format_func=lambda x: x.split(" ", 1)[1]
    )
    
    # API Key status
    st.sidebar.subheader("🔑 API Configuration")
    groq_key = get_groq_api_key()
    if groq_key:
        st.sidebar.success("✅ Groq API Key configured")
    else:
        st.sidebar.warning("⚠️ Groq API Key required for AI responses")
        st.sidebar.markdown("[Get free API key from Groq](https://groq.com)")
    
    # Model status
    st.sidebar.subheader("🤖 AI Models")
    if st.session_state.model:
        st.sidebar.success("✅ Main model loaded")
    else:
        st.sidebar.error("❌ Main model failed to load")
    
    if st.session_state.multilingual_model:
        st.sidebar.success("✅ Multilingual support ready")
    else:
        st.sidebar.warning("⚠️ Limited to English only")
    
    # Data status
    st.sidebar.subheader("📊 Knowledge Base")
    st.sidebar.metric("Documents", len(st.session_state.knowledge_base))
    st.sidebar.metric("Locations", len(st.session_state.location_data))
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Quick Start")
    if page == "👤 User Chat":
        st.sidebar.markdown("""
        1. Ask questions about the campus
        2. Request directions to locations
        3. Get information from uploaded documents
        """)
    else:
        st.sidebar.markdown("""
        1. Upload PDF documents for Q&A
        2. Upload CSV/Excel with location data
        3. Monitor processing in real-time
        """)
    
    # Route to appropriate page
    if "User" in page:
        user_page()
    else:
        admin_page()

if __name__ == "__main__":
    main()
