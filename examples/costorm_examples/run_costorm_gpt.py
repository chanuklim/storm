"""
Co-STORM pipeline with support for local Ollama (default), OpenAI, or Azure models and multiple search engines.

Key environment variables when needed:
    - OPENAI_API_KEY: OpenAI API key (if --llm-provider openai)
    - AZURE_API_KEY / AZURE_API_BASE / AZURE_API_VERSION: Azure API config (if --llm-provider azure)
    - BING_SEARCH_API_KEY / SERPER_API_KEY / BRAVE_API_KEY / TAVILY_API_KEY / etc.: Retriever keys
    - OLLAMA_MODELS: Optional, directory for Ollama models (defaults to --ollama-model-dir)
    - HF_HOME: Optional, cache dir for local embedding models (defaults to --embedding-cache-dir)

Output will be structured as below
args.output_dir/
    log.json           # Log of information-seeking conversation
    report.md          # Final article generated
    instance_dump.json # Serialized run state
"""

import os
import sys
import json
import traceback
from argparse import ArgumentParser
from typing import Optional
from pathlib import Path

# Ensure repository root is on sys.path so local knowledge_storm is used even if an older package is installed.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    RunnerArgument,
    CoStormRunner,
)
from knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from knowledge_storm.collaborative_storm.modules.collaborative_storm_utils import (
    detect_language,
    translate_text,
)
from knowledge_storm.lm import LitellmModel, OpenAIModel, AzureOpenAIModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import (
    YouRM,
    BingSearch,
    BraveRM,
    SerperRM,
    DuckDuckGoSearchRM,
    TavilySearchRM,
    SearXNG,
)
from knowledge_storm.encoder import Encoder
from knowledge_storm.utils import load_api_key


def build_base_url(url: str, port: Optional[int] = None) -> str:
    """Normalize base URL and optionally append port."""
    if not url.startswith("http://") and not url.startswith("https://"):
        url = f"http://{url}"
    url = url.rstrip("/")
    if port and f":{port}" not in url.split("//", 1)[-1]:
        url = f"{url}:{port}"
    return url


def chunked_translate_report(
    translator_lm,
    text: str,
    target_lang: str = "ko",
    source_lang_hint: str = "en",
    max_chunk_chars: int = 2000,
    log_path: Optional[str] = None,
):
    """Translate long reports in chunks to reduce context-related failures."""
    if not text:
        return text
    paragraphs = text.split("\n\n")
    # Pair headings with the following paragraph to preserve structure.
    paired_paragraphs = []
    skip_next = False
    for idx, para in enumerate(paragraphs):
        if skip_next:
            skip_next = False
            continue
        if para.strip().startswith("#") and idx + 1 < len(paragraphs):
            paired_paragraphs.append(f"{para}\n\n{paragraphs[idx + 1]}")
            skip_next = True
        else:
            paired_paragraphs.append(para)

    chunks = []
    buffer = ""
    for para in paired_paragraphs:
        # If a single paragraph is too large, break it up by character count.
        if len(para) > max_chunk_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            for i in range(0, len(para), max_chunk_chars):
                chunks.append(para[i : i + max_chunk_chars])
            continue
        if len(buffer) + len(para) + 2 <= max_chunk_chars:
            buffer = para if not buffer else f"{buffer}\n\n{para}"
        else:
            chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)

    translated_chunks = []
    log_entries = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            log_entries.append({"chunk_index": idx, "status": "skipped_empty"})
            continue
        prompt = (
            f"Translate the following Markdown to {target_lang}. "
            "Keep the Markdown structure exactly and do not add any prefixes, notes, or explanations. "
            "Only return the translated Markdown content.\n"
            f"Source language hint: {source_lang_hint}.\n\n"
            f"Text:\n{chunk}\n\nTranslation:"
        )
        translated = translator_lm(prompt)[0].strip()
        status = "translated"
        if not translated:
            translated = chunk
            status = "fallback_original"
        # Strip common artifacts the model might add.
        translated = translated.replace("Translated (ko):", "").strip()
        translated_chunks.append(translated)
        log_entries.append(
            {
                "chunk_index": idx,
                "input_chars": len(chunk),
                "output_chars": len(translated),
                "status": status,
            }
        )
    if log_path:
        try:
            with open(log_path, "w") as lf:
                json.dump(log_entries, lf, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return "\n\n".join(translated_chunks)


def main(args):
    load_api_key(toml_file_path=args.secrets_file)
    lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()
    if args.llm_provider == "ollama" and args.ollama_model_dir:
        os.environ.setdefault("OLLAMA_MODELS", args.ollama_model_dir)

    if args.encoder_type == "hf_local" and args.embedding_cache_dir:
        os.environ.setdefault("HF_HOME", args.embedding_cache_dir)

    embedding_base_url = (
        build_base_url(args.embedding_base_url, args.embedding_port)
        if args.encoder_type == "ollama"
        else None
    )
    encoder_device = None if args.embedding_device == "auto" else args.embedding_device
    encoder = Encoder(
        encoder_type=args.encoder_type,
        api_base=embedding_base_url,
        model=args.embedding_model,
        device=encoder_device,
        cache_dir=args.embedding_cache_dir if args.encoder_type == "hf_local" else None,
    )

    llm_provider = args.llm_provider.lower()
    if llm_provider == "ollama":
        llm_base_url = build_base_url(args.llm_url, args.llm_port)
        model_name = args.llm_model
        if not model_name.startswith("ollama/"):
            model_name = f"ollama/{model_name}"
        ollama_kwargs = {
            "base_url": llm_base_url,
            "temperature": args.llm_temperature,
            "top_p": args.llm_top_p,
            "model_type": "chat",
        }

        def build_lm(max_tokens: int):
            return LitellmModel(
                model=model_name,
                max_tokens=max_tokens,
                **ollama_kwargs,
            )

    elif llm_provider == "openai":
        openai_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_provider": "openai",
            "temperature": args.llm_temperature,
            "top_p": args.llm_top_p,
            "api_base": None,
        }
        ModelClass = OpenAIModel
        gpt_4o_model_name = "gpt-4o"

        def build_lm(max_tokens: int):
            return ModelClass(
                model=gpt_4o_model_name, max_tokens=max_tokens, **openai_kwargs
            )

    elif llm_provider == "azure":
        openai_kwargs = {
            "api_key": os.getenv("AZURE_API_KEY"),
            "temperature": args.llm_temperature,
            "top_p": args.llm_top_p,
            "api_base": os.getenv("AZURE_API_BASE"),
            "api_version": os.getenv("AZURE_API_VERSION"),
        }
        ModelClass = AzureOpenAIModel
        gpt_4o_model_name = "gpt-4o"

        def build_lm(max_tokens: int):
            return ModelClass(
                model=gpt_4o_model_name, max_tokens=max_tokens, **openai_kwargs
            )

    else:
        raise ValueError(
            f'Invalid llm provider: {args.llm_provider}. Choose either "ollama", "openai", or "azure".'
        )

    # STORM is a LM system so different components can be powered by different models.
    question_answering_lm = build_lm(1000)
    discourse_manage_lm = build_lm(500)
    utterance_polishing_lm = build_lm(2000)
    warmstart_outline_gen_lm = build_lm(500)
    question_asking_lm = build_lm(300)
    knowledge_base_lm = build_lm(1000)

    # Use a separate translator model so user-facing translations don't pollute LM history
    translator_lm = build_lm(500)

    lm_config.set_question_answering_lm(question_answering_lm)
    lm_config.set_discourse_manage_lm(discourse_manage_lm)
    lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
    lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
    lm_config.set_question_asking_lm(question_asking_lm)
    lm_config.set_knowledge_base_lm(knowledge_base_lm)

    topic_raw = input("Topic: ")
    user_lang = detect_language(topic_raw)
    topic = (
        translate_text(translator_lm, topic_raw, target_lang="en", source_lang_hint="ko")
        if user_lang == "ko"
        else topic_raw
    )
    runner_argument = RunnerArgument(
        topic=topic,
        language=user_lang,
        retrieve_top_k=args.retrieve_top_k,
        max_search_queries=args.max_search_queries,
        total_conv_turn=args.total_conv_turn,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=args.moderator_override_N_consecutive_answering_turn,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
    )
    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = (
        LocalConsolePrintCallBackHandler() if args.enable_log_print else None
    )

    # Co-STORM is a knowledge curation system which consumes information from the retrieval module.
    # Currently, the information source is the Internet and we use search engine API as the retrieval module.
    match args.retriever:
        case "bing":
            rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "you":
            rm = YouRM(
                ydc_api_key=os.getenv("YDC_API_KEY"), k=runner_argument.retrieve_top_k
            )
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(
                k=runner_argument.retrieve_top_k, safe_search="On", region="us-en"
            )
        case "serper":
            rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=runner_argument.retrieve_top_k,
                include_raw_content=True,
            )
        case "searxng":
            rm = SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. Choose either "bing", "you", "brave", "duckduckgo", "serper", "tavily", or "searxng"'
            )

    os.makedirs(args.output_dir, exist_ok=True)
    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=rm,
        encoder=encoder,
        callback_handler=callback_handler,
    )

    article = None
    instance_copy = None
    log_dump = None
    error_payload = None
    error_exc = None

    try:
        # warm start the system
        costorm_runner.warm_start()

        # Below is an example of how users may interact with Co-STORM to seek information together
        # In actual deployment, we suggest allowing the user to decide whether to observe the agent utterance or inject a turn

        # observing Co-STORM LLM agent utterance for 5 turns
        for _ in range(1):
            conv_turn = costorm_runner.step()
            utter_to_show = (
                translate_text(translator_lm, conv_turn.utterance, target_lang="ko")
                if user_lang == "ko"
                else conv_turn.utterance
            )
            print(f"**{conv_turn.role}**: {utter_to_show}\n")

        # active engaging by injecting your utterance
        your_utterance = input("Your utterance: ")
        if user_lang == "ko":
            your_utterance = translate_text(
                translator_lm, your_utterance, target_lang="en", source_lang_hint="ko"
            )
        costorm_runner.step(user_utterance=your_utterance)

        # continue observing
        conv_turn = costorm_runner.step()
        utter_to_show = (
            translate_text(translator_lm, conv_turn.utterance, target_lang="ko")
            if user_lang == "ko"
            else conv_turn.utterance
        )
        print(f"**{conv_turn.role}**: {utter_to_show}\n")

        # generate report
        costorm_runner.knowledge_base.reorganize()
        article = costorm_runner.generate_report()
    except Exception as exc:
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"Run failed: {exc}")
        error_exc = exc
    finally:
        try:
            instance_copy = costorm_runner.to_dict()
        except Exception as e:
            instance_copy = instance_copy or {"error": f"instance_dump_failed: {e}"}
        try:
            log_dump = costorm_runner.dump_logging_and_reset()
        except Exception as e:
            log_dump = log_dump or {"error": f"log_dump_failed: {e}"}

        # Save artifacts if available
        if article is not None:
            # If the pipeline ran in Korean, the generated article is already Korean.
            if user_lang == "ko":
                article_kr = article
                with open(os.path.join(args.output_dir, "report_kr.md"), "w") as f:
                    f.write(article_kr)
            else:
                # Default: English generation
                with open(os.path.join(args.output_dir, "report_eng.md"), "w") as f:
                    f.write(article)

        if instance_copy is not None:
            with open(os.path.join(args.output_dir, "instance_dump.json"), "w") as f:
                json.dump(instance_copy, f, indent=2)

        if log_dump is not None:
            # Attach run configuration to help post-run analysis (including language).
            try:
                log_dump["runner_argument"] = runner_argument.to_dict()
            except Exception:
                pass
            with open(os.path.join(args.output_dir, "log.json"), "w") as f:
                json.dump(log_dump, f, indent=2)

        if error_payload is not None:
            with open(os.path.join(args.output_dir, "error.json"), "w") as f:
                json.dump(error_payload, f, indent=2)
            # Re-raise to surface failure after saving artifacts
            if error_exc is not None:
                raise error_exc


if __name__ == "__main__":
    parser = ArgumentParser()
    # global arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/co-storm",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["ollama", "openai", "azure"],
        default="ollama",
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-oss:120b",
        help="Model name for the selected LLM provider (for Ollama, omit the 'ollama/' prefix).",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost",
        help="Base URL for the LLM service (used for Ollama).",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=11434,
        help="Port for the LLM service (used for Ollama).",
    )
    parser.add_argument(
        "--ollama-model-dir",
        type=str,
        default="/data/ollama/models",
        help="Directory where Ollama should store models.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--llm-top-p",
        type=float,
        default=0.9,
        help="Top-p for nucleus sampling.",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        choices=["hf_local", "ollama", "openai", "azure"],
        default="hf_local",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="/data/models/nvidia-llama-embed-nemotron-8b",
        help="Embedding model name or local path.",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default="http://localhost",
        help="Base URL for embedding service when encoder-type is ollama.",
    )
    parser.add_argument(
        "--embedding-port",
        type=int,
        default=11434,
        help="Port for embedding service when encoder-type is ollama.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default="auto",
        help="Device for local embeddings (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=str,
        default="/data/models",
        help="Cache directory / HF_HOME for local embedding models.",
    )
    parser.add_argument(
        "--secrets-file",
        type=str,
        default="/data/coscientist/secrets.toml",
        help="Path to secrets.toml for API keys.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"],
        default="duckduckgo",
        help="The search engine API to use for retrieving information.",
    )
    # hyperparameters for co-storm
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=10,
        help="Retrieve top k results for each query in retriever.",
    )
    parser.add_argument(
        "--max_search_queries",
        type=int,
        default=2,
        help="Maximum number of search queries to consider for each question.",
    )
    parser.add_argument(
        "--total_conv_turn",
        type=int,
        default=20,
        help="Maximum number of turns in conversation.",
    )
    parser.add_argument(
        "--max_search_thread",
        type=int,
        default=5,
        help="Maximum number of parallel threads for retriever.",
    )
    parser.add_argument(
        "--max_search_queries_per_turn",
        type=int,
        default=3,
        help="Maximum number of search queries to consider in each turn.",
    )
    parser.add_argument(
        "--warmstart_max_num_experts",
        type=int,
        default=3,
        help="Max number of experts in perspective-guided QA during warm start.",
    )
    parser.add_argument(
        "--warmstart_max_turn_per_experts",
        type=int,
        default=2,
        help="Max number of turns per perspective during warm start.",
    )
    parser.add_argument(
        "--warmstart_max_thread",
        type=int,
        default=3,
        help="Max number of threads for parallel perspective-guided QA during warm start.",
    )
    parser.add_argument(
        "--max_thread_num",
        type=int,
        default=10,
        help=(
            "Maximum number of threads to use. "
            "Consider reducing it if you keep getting 'Exceed rate limit' errors when calling the LM API."
        ),
    )
    parser.add_argument(
        "--max_num_round_table_experts",
        type=int,
        default=2,
        help="Max number of active experts in round table discussion.",
    )
    parser.add_argument(
        "--moderator_override_N_consecutive_answering_turn",
        type=int,
        default=3,
        help=(
            "Number of consecutive expert answering turns before the moderator overrides the conversation."
        ),
    )
    parser.add_argument(
        "--node_expansion_trigger_count",
        type=int,
        default=10,
        help="Trigger node expansion for nodes that contain more than N snippets.",
    )

    # Boolean flags
    parser.add_argument(
        "--enable_log_print",
        action="store_true",
        help="If set, enable console log print.",
    )

    main(parser.parse_args())
