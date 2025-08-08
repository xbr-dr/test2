import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from typing import List, Dict, Optional

# Set page config
st.set_page_config(
    page_title="Campus Assistant",
    page_icon="🏫",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = []
    if 'location_data' not in st.session_state:
        st.session_state.location_data = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'search_index' not in st.session_state:
        st.session_state.search_index = None

init_session_state()

# Load dependencies with error handling
@st.cache_resource
def load_dependencies():
    """Load optional dependencies with graceful fallback"""
    deps = {
        'pdf': False,
        'transformers': False,
        'faiss': False,
        'folium': False,
        'langdetect': False,
        'groq': False
    }
    
    try:
        import PyPDF2
        deps['pdf'] = True
    except ImportError:
        st.warning("PDF support not available")
    
    try:
        from sentence_transformers import SentenceTransformer
        deps['transformers'] = True
    except ImportError:
        st.warning("AI search not available - using basic text matching")
    
    try:
        import faiss
        deps['faiss'] = True
    except ImportError:
        st.warning("Advanced search not available")
    
    try:
        import folium
        from streamlit_folium import st_folium
        deps['folium'] = True
    except ImportError:
        st.warning("Maps not available")
    
    try:
        from langdetect import detect
        deps['langdetect'] = True
    except ImportError:
        pass
    
    try:
        from groq import Groq
        deps['groq'] = True
    except ImportError:
        st.warning("AI chat not available - install groq package")
    
    return deps

@st.cache_resource
def load_ai_model():
    """Load AI model if available"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Could not load AI model: {e}")
        return None

def get_api_key():
    """Get API key from secrets or user input"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return st.sidebar.text_input("Groq API Key", type="password")

def extract_pdf_text(pdf_file):
    """Extract text from PDF with fallback"""
    deps = load_dependencies()
    if not deps['pdf']:
        return "PDF processing not available. Please install PyPDF2."
    
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {e}"

def process_text(text: str) -> List[str]:
    """Split text into chunks"""
    if not text:
        return []
    
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+|\n\n+', text)
    
    # Clean and filter
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if 20 <= len(sentence) <= 500:
            chunks.append(sentence)
    
    return chunks

def extract_locations(df: pd.DataFrame) -> List[Dict]:
    """Extract location data from DataFrame"""
    locations = []
    
    if df.empty:
        return locations
    
    # Find relevant columns (case insensitive)
    cols = {col.lower(): col for col in df.columns}
    
    name_col = None
    lat_col = None
    lon_col = None
    desc_col = None
    
    # Look for name column
    for variant in ['name', 'location', 'place', 'building']:
        if variant in cols:
            name_col = cols[variant]
            break
    
    # Look for latitude
    for variant in ['latitude', 'lat', 'y']:
        if variant in cols:
            lat_col = cols[variant]
            break
    
    # Look for longitude
    for variant in ['longitude', 'lon', 'lng', 'x']:
        if variant in cols:
            lon_col = cols[variant]
            break
    
    # Look for description
    for variant in ['description', 'desc', 'info']:
        if variant in cols:
            desc_col = cols[variant]
            break
    
    # Extract if we have required columns
    if name_col and lat_col and lon_col:
        for _, row in df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    location = {
                        'name': str(row[name_col]).strip(),
                        'latitude': lat,
                        'longitude': lon,
                        'description': str(row[desc_col]).strip() if desc_col else ""
                    }
                    locations.append(location)
            except (ValueError, TypeError):
                continue
    
    return locations

def basic_search(query: str, texts: List[str], k: int = 3) -> List[str]:
    """Basic keyword search when AI is not available"""
    if not query or not texts:
        return []
    
    query_words = query.lower().split()
    
    # Score texts based on keyword matches
    scored_texts = []
    for text in texts:
        text_lower = text.lower()
        score = sum(word in text_lower for word in query_words)
        if score > 0:
            scored_texts.append((score, text))
    
    # Sort by score and return top k
    scored_texts.sort(reverse=True, key=lambda x: x[0])
    return [text for _, text in scored_texts[:k]]

def ai_search(query: str, texts: List[str], model, k: int = 3) -> List[str]:
    """AI-powered search if available"""
    deps = load_dependencies()
    
    if not deps['transformers'] or not deps['faiss'] or not model:
        return basic_search(query, texts, k)
    
    try:
        import faiss
        
        # Create embeddings
        embeddings = model.encode(texts)
        query_embedding = model.encode([query])
        
        # Create index
        index = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Normalize
        faiss.normalize_L2(embeddings)
        faiss.normalize_L2(query_embedding)
        
        # Add and search
        index.add(embeddings.astype('float32'))
        _, indices = index.search(query_embedding.astype('float32'), k)
        
        return [texts[i] for i in indices[0] if i < len(texts)]
    except:
        return basic_search(query, texts, k)

def find_location(query: str, locations: List[Dict]) -> Optional[Dict]:
    """Find location mentioned in query"""
    query_lower = query.lower()
    
    for location in locations:
        if location['name'].lower() in query_lower:
            return location
    
    return None

def generate_response(query: str, context: List[str], api_key: str) -> str:
    """Generate AI response or fallback"""
    deps = load_dependencies()
    
    if not deps['groq'] or not api_key:
        # Fallback response
        if context:
            return f"Based on the available information: {context[0][:200]}..."
        else:
            return "I don't have specific information about that. Please upload relevant documents or check if your question relates to available campus locations."
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        context_text = "\n".join(context[:2]) if context else "No specific context available."
        
        prompt = f"""Answer this campus question based on the context provided. Be helpful and concise.

Context: {context_text}

Question: {query}

Answer:"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I couldn't generate a response. Error: {e}"

def show_map(location: Dict):
    """Show map if folium is available"""
    deps = load_dependencies()
    
    if not deps['folium']:
        st.write(f"📍 **{location['name']}**")
        st.write(f"Coordinates: {location['latitude']}, {location['longitude']}")
        if location['description']:
            st.write(f"Description: {location['description']}")
        return
    
    try:
        import folium
        from streamlit_folium import st_folium
        
        m = folium.Map(
            location=[location['latitude'], location['longitude']], 
            zoom_start=16
        )
        
        folium.Marker(
            [location['latitude'], location['longitude']],
            popup=f"<b>{location['name']}</b><br>{location['description']}",
            tooltip=location['name']
        ).add_to(m)
        
        st_folium(m, height=300)
    except Exception as e:
        st.error(f"Map error: {e}")

def admin_page():
    """Admin interface"""
    st.title("🔧 Admin Panel")
    
    # File uploader
    files = st.file_uploader(
        "Upload Files",
        type=['pdf', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if files:
        progress = st.progress(0)
        
        all_texts = []
        all_locations = []
        
        for i, file in enumerate(files):
            st.write(f"Processing {file.name}...")
            
            try:
                if file.type == "application/pdf":
                    text = extract_pdf_text(file)
                    if text and not text.startswith("Error"):
                        chunks = process_text(text)
                        all_texts.extend(chunks)
                        st.success(f"✅ {file.name}: {len(chunks)} text chunks")
                
                elif file.name.endswith(('.csv', '.xlsx', '.xls')):
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    # Extract locations
                    locations = extract_locations(df)
                    all_locations.extend(locations)
                    
                    # Also process as text
                    text = df.to_string()
                    chunks = process_text(text)
                    all_texts.extend(chunks)
                    
                    st.success(f"✅ {file.name}: {len(locations)} locations, {len(chunks)} text chunks")
                    
                    if not df.empty:
                        st.dataframe(df.head())
            
            except Exception as e:
                st.error(f"❌ {file.name}: {e}")
            
            progress.progress((i + 1) / len(files))
        
        # Update session state
        if all_texts:
            st.session_state.knowledge_base.extend(all_texts)
        if all_locations:
            st.session_state.location_data.extend(all_locations)
        
        st.success("Processing complete!")
    
    # Statistics
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Text Chunks", len(st.session_state.knowledge_base))
    with col2:
        st.metric("Locations", len(st.session_state.location_data))
    
    # Show locations
    if st.session_state.location_data:
        st.subheader("📍 Locations")
        df = pd.DataFrame(st.session_state.location_data)
        st.dataframe(df)
    
    # Clear data
    if st.button("Clear All Data"):
        st.session_state.knowledge_base = []
        st.session_state.location_data = []
        st.session_state.chat_history = []
        st.rerun()

def user_page():
    """User chat interface"""
    st.title("💬 Campus Assistant")
    
    # Load model if available
    if not st.session_state.models_loaded:
        model = load_ai_model()
        st.session_state.search_model = model
        st.session_state.models_loaded = True
    
    # Show chat history
    for user_msg, bot_msg, location in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        
        with st.chat_message("assistant"):
            st.write(bot_msg)
            if location:
                show_map(location)
    
    # Chat input
    if prompt := st.chat_input("Ask about the campus..."):
        # Search knowledge base
        if st.session_state.knowledge_base:
            if hasattr(st.session_state, 'search_model') and st.session_state.search_model:
                context = ai_search(prompt, st.session_state.knowledge_base, st.session_state.search_model)
            else:
                context = basic_search(prompt, st.session_state.knowledge_base)
        else:
            context = []
        
        # Find location
        location = find_location(prompt, st.session_state.location_data)
        
        # Generate response
        api_key = get_api_key()
        response = generate_response(prompt, context, api_key)
        
        # Add to history
        st.session_state.chat_history.append((prompt, response, location))
        st.rerun()

def main():
    """Main app"""
    # Sidebar
    st.sidebar.title("🏫 Campus Assistant")
    
    # Check dependencies
    deps = load_dependencies()
    st.sidebar.subheader("System Status")
    
    status_items = [
        ("PDF Processing", deps['pdf']),
        ("AI Search", deps['transformers'] and deps['faiss']),
        ("Maps", deps['folium']),
        ("AI Chat", deps['groq'])
    ]
    
    for item, status in status_items:
        if status:
            st.sidebar.success(f"✅ {item}")
        else:
            st.sidebar.warning(f"⚠️ {item}")
    
    # Page selection
    page = st.sidebar.selectbox("Choose Page", ["User Chat", "Admin Panel"])
    
    # API key status
    api_key = get_api_key()
    if api_key:
        st.sidebar.success("🔑 API Key Set")
    else:
        st.sidebar.warning("🔑 API Key Missing")
    
    # Routes
    if page == "Admin Panel":
        admin_page()
    else:
        user_page()

if __name__ == "__main__":
    main()
