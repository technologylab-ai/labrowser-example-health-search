# How People Search for Health Information

Sample data and analysis code for a [LaBrowser](https://github.com/technologylab-ai/labrowser) example study.

20 participants researched intermittent fasting using Google Search, health websites, and ChatGPT. LaBrowser captured every search query, click, page visit, and AI prompt as structured event data. This repository contains the exported data and a Jupyter notebook that reproduces the analysis shown on the [showcase page](https://labrowser.technologylab.ai/examples/health-search).

## What's in the data

| File | Records | Description |
|------|---------|-------------|
| `data/events.json` | ~3,300 events | Raw event stream: NAVIGATE, PAGE_LOADED, CLICK, SCROLL, INPUT_SUBMIT, TAB_* |
| `data/google_search_v1.json` | ~116 sessions | Derived Google search sessions with queries, result clicks, and dwell times |
| `data/chatgpt_session_v1.json` | ~7 conversations | Derived ChatGPT conversations with prompts and response detection |
| `data/study_config.json` | 1 config | Study configuration (allowed domains, capture rules, parsers) |

All files use the exact export format from the LaBrowser Study Console.

## Key findings

- **Three research strategies emerged**: deep divers (few queries, long reading), wide scanners (many queries, quick skimming), and AI-assisted (ChatGPT + targeted searches)
- **ChatGPT users issued fewer search queries** (5.8 vs 7.4) and visited fewer sources (8.2 vs 12.1), suggesting AI partially substitutes for traditional web searching
- **Academic sources (PubMed, Harvard) had the longest dwell times** (72-84s) despite getting fewer clicks than popular health sites like Healthline (52s)
- **30% of participants used ChatGPT** alongside Google Search, primarily for summaries and safety-related questions

## Quick start

```bash
# Clone and set up
git clone https://github.com/technologylab-ai/labrowser-example-health-search
cd labrowser-example-health-search
pip install -r requirements.txt

# Run the analysis notebook
jupyter lab analysis.ipynb

# Or run the script version
python analysis.py
```

## Requirements

- Python 3.10+
- See `requirements.txt` for packages (pandas, numpy, matplotlib, seaborn, plotly, jupyter)

## Data format reference

### Events (`events.json`)

Each event has:

```json
{
    "id": "uuid",
    "session_id": "uuid",
    "tab_id": "uuid",
    "timestamp_utc": "2025-11-15T14:23:01.456Z",
    "event_type": "NAVIGATE|PAGE_LOADED|CLICK|SCROLL|INPUT_SUBMIT|TAB_OPENED|...",
    "page_id": "uuid",
    "url": "https://...",
    "payload": { }
}
```

Event types and their payloads:
- **NAVIGATE**: `from_url`, `to_url`, `trigger` (link_click, address_bar, etc.)
- **PAGE_LOADED**: `url`, `title`, `hostname`, `load_time_ms`, viewport dimensions
- **CLICK**: `x`, `y`, `element_tag`, `element_text_snippet`, `css_selector`
- **SCROLL**: `scroll_top`, `viewport_height`, `document_height`
- **INPUT_SUBMIT**: `field_role`, `service`, `text`, `url`, `submit_method`

### Derived sessions (`google_search_v1.json`, `chatgpt_session_v1.json`)

Each derived session has:

```json
{
    "id": "uuid",
    "study_id": "uuid",
    "session_id": "uuid",
    "parser_id": "google_search_v1",
    "type": "google_search",
    "start_time": "2025-11-15T14:23:01.456Z",
    "end_time": "2025-11-15T14:28:15.789Z",
    "payload": { }
}
```

## About LaBrowser

[LaBrowser](https://github.com/technologylab-ai/labrowser) is a dedicated research browser for behavioral studies. Participants use it instead of their normal browser — everything inside is logged as structured, typed events. Everything outside is untouched.

Learn more at [labrowser.technologylab.ai](https://labrowser.technologylab.ai).
