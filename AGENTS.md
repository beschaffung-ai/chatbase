# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
Single-service Streamlit app (Python) — a Chatbase Analytics Dashboard for analyzing chatbot conversation logs. No database, no Docker, no additional backend services. See `README.md` for feature details.

### Running the app
```bash
python3 -m streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
```
The app runs on port 8501. It requires `.streamlit/secrets.toml` for authentication (git-ignored). Minimal working secrets file:
```toml
[passwords]
testuser = "testpass123"
session_secret = "dev-session-secret-key"
```

### Key caveats
- Use `python3` not `python` — the environment does not have `python` symlinked.
- `python3-dev` and `build-essential` are needed to build `hdbscan` from source (C extension compilation). These are system packages, not pip packages.
- NLTK stopwords corpus is auto-downloaded on first run; no manual step needed.
- The OpenAI-based features (embedding clustering, AI topic modeling) require an `[openai]` section in `secrets.toml` with `api_key`. These features are optional; all core analytics work without it.
- A test CSV file at `/workspace/test_data.csv` can be used for quick functional testing via the file upload widget.

### Linting
No linting tool is configured in this project. Use `python3 -m py_compile <file>` for basic syntax checks. Consider using `ruff` if a linter is needed.

### Tests
No automated test suite exists. Manual testing via the Streamlit UI is the primary testing method.
