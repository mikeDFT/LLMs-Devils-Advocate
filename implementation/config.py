from pathlib import Path

# paths

PROJECT_ROOT = Path(__file__).parent.parent
IMPLEMENTATION_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
SEP_ARTICLES_DIR = DATA_DIR / "sep_articles"
WIKI_ARTICLES_DIR = DATA_DIR / "wiki_articles"
WIKI_LISTS_DIR = DATA_DIR / "wiki_lists"
SFT_DATASET_PATH = DATA_DIR / "sft_dataset.jsonl"
PREFERENCE_DATASET_PATH = DATA_DIR / "preference_dataset.jsonl"
DEBATESUM_CSV_PATH = PROJECT_ROOT / "DebateSumV3.csv"

SFT_ADAPTER_DIR = IMPLEMENTATION_ROOT / "sft_adapter"
DPO_ADAPTER_DIR = IMPLEMENTATION_ROOT / "dpo_adapter"
GGUF_OUTPUT_DIR = IMPLEMENTATION_ROOT / "gguf_output"
FALLACY_MODEL_DIR = IMPLEMENTATION_ROOT / "fallacy_model"
RAG_DB_DIR = IMPLEMENTATION_ROOT / "debate_rag_db"
EVAL_OUTPUT_DIR = IMPLEMENTATION_ROOT / "evaluation_results"

# model

BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
MAX_PROMPT_LENGTH = 1280
MAX_SEQ_LENGTH = 2048
RANDOM_SEED = 1230946767

# LoRA and SFT

LORA_RANK = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SFT_LEARNING_RATE = 2e-4
SFT_BATCH_SIZE = 2
SFT_GRADIENT_ACCUMULATION = 4
SFT_EPOCHS = 2

# DPO

DPO_BETA = 0.1
DPO_LEARNING_RATE = 1e-6
DPO_BATCH_SIZE = 2
DPO_GRADIENT_ACCUMULATION = 4
DPO_MAX_STEPS = 160
MIN_JUDGE_SCORE = 70  # RLAIF: discard pairs where chosen response scores below this

# embeddings and RAG

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v2-moe"
RAG_TABLE_NAME = "debate_knowledge"
RAG_TOP_K = 3
CHUNK_SIZE = 512
CHUNK_SIMILARITY_THRESHOLD = 0.5

# Reranker

RERANKER_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
RERANKER_TOP_K = 3
RERANKER_SCORE_THRESHOLD = 0.1

# fallacy detector

FALLACY_BASE_MODEL = "answerdotai/ModernBERT-base"
FALLACY_DATASET = "tasksource/logical-fallacy"
FALLACY_NUM_LABELS = 14  # 13 fallacy types + "no fallacy"
FALLACY_LABELS = [
    "no_fallacy",
    "ad_hominem",
    "ad_populum",
    "appeal_to_emotion",
    "circular_reasoning",
    "equivocation",
    "fallacy_of_credibility",
    "fallacy_of_extension",
    "fallacy_of_logic",
    "fallacy_of_relevance",
    "false_causality",
    "false_dilemma",
    "intentional",
    "faulty_generalization",
]
FALLACY_CONFIDENCE_THRESHOLD = 0.45

# LMStudio

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MAIN_MODEL = "qwen3-4b-dpo"
LMSTUDIO_UTILITY_MODEL = "qwen/qwen3-4b-2507"
MAX_TOOL_CALLS_PER_TURN = 3
CONTEXT_WINDOW_EXCHANGES = 6

# Query reformulation

REFORMULATION_TEMPERATURE = 0.2
REFORMULATION_MAX_TOKENS = 200
REFORMULATION_NUM_QUERIES = 3

# sys prompt

SYSTEM_PROMPT = (
    "You are the Devil's Advocate. Your sole purpose is to oppose, attack, and dismantle "
    "every claim the user makes. Expose weak evidence, logical fallacies, and flawed premises. "
    "Never concede, agree, or validate. Never say 'good point', 'you're right', or anything sycophantic. "
    "Cite specific names, dates, and data to destroy their position."
)
