# Implementation guide

1. Activate `.venv` and install `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env`, set a real MySQL `DATABASE_URL`, then run `python -m database.initialize`.
3. Start llama.cpp locally, for example: `llama-server -m models\\qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 9931 -t 4 -c 2048`.
4. Start the API with `uvicorn api.app:app --host 127.0.0.1 --port 8000` and dashboard with `streamlit run dashboard.py`.
5. Run tests with `pytest -q`.

Do not bind local inference or API services to public interfaces without deliberate security review.
