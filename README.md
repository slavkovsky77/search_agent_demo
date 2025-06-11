# 🔍 AI Internet Search Agent

An AI-powered agent that autonomously searches for and downloads content

## Features

- 🤖 **Natural language queries** - Just ask for what you want
- 🖼️ **Image search** - Find and download photos
- 📄 **Article download** - Get articles from any website
- 🌐 **Web GUI** - Beautiful Streamlit interface with real-time logs
- 🐋 **Dockerized** - One-command deployment

## Quick Start

### With Docker (Recommended)

```bash
# Start the web interface
docker compose up streamlit

# Access at http://localhost:8544
```

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Start SearXNG
docker run -d -p 8080:8080 searxng/searxng

# Run web GUI
streamlit run streamlit_app.py --server.port=8544

# Or use CLI
python -m src.agent_search_v2 "Find 3 photos of cats"
```

## Example Requests

- `Find 3 photos of zebras`
- `Download 2 random Wikipedia articles`
- `Download the front page of https://news.ycombinator.com/`
- `Download 2 articles about AI from nasa.gov`


## Architecture

- **Search**: SearXNG (self-hosted)
- **AI**: OpenRouter API (GPT-4, Claude, etc.)
- **Storage**: Local filesystem
- **Web UI**: Streamlit with real-time logging

---

*Just ask for what you want - the AI figures out the rest.*