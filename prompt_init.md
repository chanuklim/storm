Ollama + Nemotron Embed migration plan

1) Baseline review
- Read `examples/costorm_examples/run_costorm_gpt.py` to map all six LM roles now built with `OpenAIModel`/Azure and confirm env vars consumed.
- Trace where `Encoder()` is constructed (e.g., `CoStormRunner` and agents) and note it only supports `ENCODER_API_TYPE` openai/azure today.

2) Swap LLM to Ollama gpt-oss-120b
- Add or reuse an Ollama-compatible chat wrapper in `knowledge_storm/lm.py` (matching `OpenAIModel` interface; use `OllamaClient`/`dspy.OllamaLocal`) with base_url f"{url}:{port}" and usage tracking.
- Extend `CollaborativeStormLMConfigs` to accept `lm_type="ollama"` with defaults model=`gpt-oss-120b`, url=`http://localhost`, port=11434, and temperature/top_p passthrough.
- Update `run_costorm_gpt.py` args to select provider (`--llm-provider`), model, url, port, temperature, top_p, and an optional `--ollama-model-dir` that sets `os.environ["OLLAMA_MODELS"]` (default `/data/ollama/models`).
- Instantiate the six Co-STORM LMs via the Ollama wrapper when provider==ollama; keep OpenAI/Azure paths intact for fallback.

3) Update embeddings to nvidia/llama-embed-nemotron-8b
- Expand `knowledge_storm/encoder.py` to support a local HF/ollama encoder type (e.g., `hf_local` or `ollama_embed`) that accepts model name/path, device, cache_dir.
- Plumb encoder config through `CoStormRunner` so an `Encoder` instance or params can be passed instead of always `Encoder()`, and propagate to knowledge base/agents.
- Add CLI/env flags in `run_costorm_gpt.py` for encoder type, embedding model, embedding_base_url/port (if using Ollama), device, and cache dir; default to `/data/models/nvidia-llama-embed-nemotron-8b`.

4) Model acquisition steps (manual)
- LLM: `export OLLAMA_MODELS=/data/ollama/models; ollama pull gpt-oss-120b`; ensure `ollama serve` runs with the same OLLAMA_MODELS.
- Embedding options:
  - HF local: `export HF_HOME=/data/models; huggingface-cli download nvidia/llama-embed-nemotron-8b --local-dir /data/models/nvidia-llama-embed-nemotron-8b --local-dir-use-symlinks False`.
  - Ollama embed: `export OLLAMA_MODELS=/data/models; ollama pull nvidia/llama-embed-nemotron-8b` (run a service or switch OLLAMA_MODELS when calling the embed endpoint).

5) Smoke validation and docs
- With services up and models present, run a minimal topic: `python examples/costorm_examples/run_costorm_gpt.py --retriever duckduckgo --llm-provider ollama --llm-model gpt-oss-120b --llm-url http://localhost --llm-port 11434 --encoder-type hf_local --embedding-model /data/models/nvidia-llama-embed-nemotron-8b --enable_log_print`.
- Verify outputs and local calls; tune timeouts/max_tokens if needed. Update script header/README to document the new local mode, required env vars (OLLAMA_MODELS, HF_HOME/ENCODER_API_TYPE), and download steps.
