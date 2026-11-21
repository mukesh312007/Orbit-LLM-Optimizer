# Orbit local-first operation

Orbit runs inference only on the local machine. The Streamlit interface binds
to `127.0.0.1:8502` and llama.cpp binds to `127.0.0.1:9931`. Prompts, answers,
models, Spider data, and experiment logs remain local. Orbit does not use a
cloud LLM API or hosted vector database.
