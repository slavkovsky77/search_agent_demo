#!/usr/bin/env python3
"""
Streamlit Web GUI for the Internet Search Agent
"""
import streamlit as st
import os
from pathlib import Path
from typing import List

# Import the agent
from src.agent_search_v2 import InternetSearchAgent
from src.models import DownloadResult

# Page config
st.set_page_config(
    page_title="AI Internet Search Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_environment():
    """Load environment variables with Streamlit secrets fallback"""
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")

    return openrouter_key, searxng_url


def initialize_agent():
    """Initialize the search agent"""
    if 'agent' not in st.session_state:
        openrouter_key, searxng_url = load_environment()

        if not openrouter_key:
            st.error("⚠️ OPENROUTER_API_KEY not found! Please set it in environment variables or Streamlit secrets.")
            st.stop()

        try:
            st.session_state.agent = InternetSearchAgent(openrouter_key, searxng_url)
            st.success("✅ Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {e}")
            st.stop()

def render_sidebar():
    """Render the sidebar with example requests"""
    st.sidebar.title("🔍 AI Search Agent")
    st.sidebar.markdown("---")

    st.sidebar.subheader("📝 Example Requests")

    examples = [
        "Find 3 photos of zebras",
        "Download 2 random Wikipedia articles",
        "Find 5 images of mountains",
        "Download 2 articles about artificial intelligence",
        "Download the front page of https://news.ycombinator.com/",
        "Download 3 articles about space from nasa.gov",
        "Find 2 photos of elephants"
    ]

    for example in examples:
        if st.sidebar.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.search_query = example
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ How it works")
    st.sidebar.markdown("""
    1. **Enter your request** in natural language
    2. **AI understands** what you want to find
    3. **Searches** using SearXNG locally
    4. **Downloads** and validates content
    5. **Scores** results for relevance
    """)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Settings")

    # Show current settings
    _, searxng_url = load_environment()
    st.sidebar.text(f"SearXNG URL: {searxng_url}")

    if st.sidebar.button("🗂️ Open Downloads Folder"):
        downloads_path = Path("downloads").absolute()
        st.sidebar.text(f"Downloads: {downloads_path}")

def render_results(results: List[DownloadResult]):
    """Render the download results"""
    if not results:
        st.warning("No results found. Try a different search query.")
        return

    st.success(f"✅ Downloaded {len(results)} items successfully!")

    # Group results by content type
    images = [r for r in results if r.mime_type and r.mime_type.startswith('image/')]
    articles = [r for r in results if r not in images]

    # Display images
    if images:
        st.subheader("🖼️ Images")

        # Create columns for images
        cols = st.columns(min(len(images), 3))

        for idx, result in enumerate(images):
            col = cols[idx % 3]

            with col:
                try:
                    # Display image
                    st.image(str(result.filepath), caption=result.title or result.filename, use_column_width=True)

                    # Image details
                    with st.expander(f"📋 Details - {result.filename}"):
                        st.write(f"**URL:** {result.url}")
                        st.write(f"**Size:** {result.size_bytes:,} bytes")
                        st.write(f"**Format:** {result.image_format or 'Unknown'}")
                        if result.image_dimensions:
                            st.write(f"**Dimensions:** {result.image_dimensions[0]}×{result.image_dimensions[1]}")
                        if result.relevance_score:
                            st.write(f"**Relevance:** {result.relevance_score:.2f}")
                        st.write(f"**Downloaded:** {result.download_time.strftime('%Y-%m-%d %H:%M:%S')}")

                        # Download button
                        with open(result.filepath, 'rb') as f:
                            st.download_button(
                                label="📥 Download",
                                data=f.read(),
                                file_name=result.filename,
                                mime=result.mime_type,
                                key=f"download_img_{idx}"
                            )

                except Exception as e:
                    st.error(f"Error displaying image {result.filename}: {e}")

    # Display articles
    if articles:
        st.subheader("📄 Articles & Webpages")

        for idx, result in enumerate(articles):
            with st.expander(f"📄 {result.title or result.filename}"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**URL:** {result.url}")
                    if result.search_description:
                        st.write(f"**Description:** {result.search_description}")

                    # Show extracted text preview
                    if result.extracted_text:
                        st.write("**Content Preview:**")
                        preview = result.extracted_text[:500] + "..." if len(result.extracted_text) > 500 else result.extracted_text
                        st.text_area("", preview, height=100, key=f"preview_{idx}", disabled=True)

                with col2:
                    st.write(f"**Size:** {result.size_bytes:,} bytes")
                    st.write(f"**Type:** {result.mime_type or 'Unknown'}")
                    if result.relevance_score:
                        st.write(f"**Relevance:** {result.relevance_score:.2f}")
                    st.write(f"**Downloaded:** {result.download_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    # Download button
                    try:
                        with open(result.filepath, 'rb') as f:
                            st.download_button(
                                label="📥 Download File",
                                data=f.read(),
                                file_name=result.filename,
                                mime=result.mime_type or "application/octet-stream",
                                key=f"download_article_{idx}"
                            )
                    except Exception as e:
                        st.error(f"Error reading file: {e}")

def main():
    """Main Streamlit app"""

    # Initialize agent
    initialize_agent()

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("🔍 AI Internet Search Agent")
    st.markdown("Enter a natural language request to search and download content from the internet.")

    # Search input
    search_query = st.text_input(
        "🎯 What would you like to find?",
        value=st.session_state.get('search_query', ''),
        placeholder="e.g., 'Find 3 photos of zebras' or 'Download 2 random Wikipedia articles'",
        key="search_input"
    )

    # Search button
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        search_button = st.button("🔍 Search & Download", type="primary", use_container_width=True)

    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.session_state.search_query = ''
        st.rerun()

    # Process search
    if search_button and search_query.strip():
        st.session_state.search_query = search_query

        with st.spinner("🤖 AI is understanding your request..."):
            try:
                # Execute the search
                results = st.session_state.agent.execute_request(search_query)

                # Store results in session state
                st.session_state.last_results = results
                st.session_state.last_query = search_query

            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                st.exception(e)
                return

    # Display results if they exist
    if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
        st.markdown("---")
        st.subheader(f"📊 Results for: '{st.session_state.last_query}'")
        render_results(st.session_state.last_results)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by AI • 🔍 SearXNG Search • 📁 Local Storage</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()