# Presentation: The Devil's Advocate

## 1. Title Slide
* **Title:** The Devil's Advocate: Adversarial Agentic Sparring for Intellectual Integrity
* **Subtitle:** Local Agentic Architecture | Hybrid RAG | Fallacy Aware
* **Authors/Team:** Ban Mihai, Bărbos Andrada, Alexe Răzvan, Beudean Carmen (Team: Trust Me)
* **Context:** Developed as a comprehensive solution for local, privacy-first logical debate sparring, integrating specialized SLMs, vector databases, and heuristic fallback mechanisms.

---

## 2. Agent Overview
* **Architecture:** The agent operates on a custom Python loop implementing a ReAct (Reason + Act) pipeline pattern. It interfaces with an OpenAI-compatible backend via LMStudio.
* **Core Pipeline Visualization:**
    1. **User Premise:** The user submits a debate claim.
    2. **Query Reformulation:** The `QueryReformulator` module expands the user's single claim into three distinct, adversarial search queries to maximize evidence retrieval breadth.
    3. **Fallacy Analysis:** The input is processed by a dedicated external classification model (`answerdotai/ModernBERT-base`) to identify any of 13 specific logical flaws.
    4. **RAG Context:** The reformulations hit a LanceDB database. The retrieved chunks are then passed through a cross-encoder reranker.
    5. **Response Generation:** The primary agent (Qwen3-4B, aligned via SFT and DPO) receives a strictly formatted prompt combining the user's claim, fallacy analysis, and RAG context to formulate its attack.
* **Context Management:** To prevent memory overload on consumer GPUs, the loop dynamically trims conversation history, strictly maintaining only the last 6 exchanges while preserving the system prompt.
* **Working Example:** 
    * *Claim:* "Everyone knows the earth is flat!"
    * *Detection:* The agent flags *ad populum*. An auxiliary Qwen model dynamically generates a 1-2 sentence explanation of exactly why this specific text is fallacious.
    * *Action:* The agent attacks the claim using retrieved spherical-earth evidence.

---

## 3. Project Overview: Goals
* **Adversarial by Design:** Unlike standard LLMs, the system prompt explicitly commands the model to: *"oppose, attack, and dismantle every claim... Never concede, agree, or validate. Never say 'good point'."*
* **Absolute Grounding:** Every counterpoint must be anchored in concrete evidence. The database is built from the Stanford Encyclopedia of Philosophy (SEP) and over 1,000 vital Wikipedia articles.
* **Local Execution & Privacy:** Built entirely for local execution on consumer hardware. The primary inference model is quantized to 4-bit GGUF format, requiring only ~3GB VRAM, ensuring 100% data privacy with zero cloud API dependencies.

---

## 4. State of the Art: Sycophancy & The Problem with Modern LLMs
* **The Helpfulness Bias:** Modern foundational models (like GPT-4 and Claude) undergo extensive Reinforcement Learning from Human Feedback (RLHF) designed to make them polite, helpful, and agreeable. This results in models that validate user opinions even when objectively false.
* **Echo Chamber Effect:** This sycophantic behavior creates a dangerous feedback loop. Users interact with AI to reinforce their pre-existing biases rather than challenge them. Over time, this degrades critical thinking and promotes misinformation.
* **Failure in Debate:** When standard LLMs attempt to debate, they frequently hedge their arguments, apologize for disagreeing, and ultimately concede to aggressive user prompting.

---

## 5. The Adversarial Landscape & Competitive Advantage
* **Comparison with existing technologies:**
    * *The Pioneers (IBM Project Debater):* Groundbreaking but relies on massive monolithic architecture, extremely high compute costs, and is restricted to enterprise cloud environments.
    * *The Grounders (Perplexity / Elicit):* Excellent at RAG and factual retrieval, but inherently designed as "helpful" search engines. They validate the user's premises by answering questions rather than actively attacking the logic.
    * *The Reasoners (OpenAI o1 / DeepSeek R1):* Possess incredibly strong logical capabilities, but completely lack a hardened adversarial debate persona or dedicated, fast fallacy detection guardrails.
* **Where Devil's Advocate Wins:** 100% Local Privacy, purely adversarial persona that refuses to break character, explicit logical fallacy guardrails running in parallel, and optimized to run entirely on a mid-range consumer GPU.

---

## 6. Logical Fallacy Detection
* **Primary Classifier:** Uses `answerdotai/ModernBERT-base` (150M parameters), chosen for its Flash Attention support and fast sequence classification.
* **Taxonomy:** Fine-tuned on 14 distinct labels (13 specific fallacy classes like *straw man*, *false causality*, *equivocation* + "no fallacy").
* **Training Data:** Utilizes the `tasksource/logical-fallacy` dataset along with propaganda detection corpuses.
* **Dynamic Contextualization:** Once a fallacy is flagged, the system calls a local utility model (Qwen 4B) with a targeted prompt to generate a concise, 2-sentence explanation of *why* the user's exact phrasing constitutes that fallacy.
* **Heuristic Fallback:** If the classifier is unavailable, the pipeline smoothly falls back to an intricate regex pattern-matching heuristic to catch common phrasing for *Ad Hominem*, *Slippery Slope*, etc.

---

## 7. RAG: The Knowledge Engine
* **Knowledge Sources:** Over 144 in-depth articles from the Stanford Encyclopedia of Philosophy and 1000+ Wikipedia articles.
* **Advanced Two-Tier Chunking:**
    1. **Structural:** Text is first split by markdown headers (e.g., `== Header ==`) to preserve the document hierarchy and metadata breadcrumbs (e.g., `Article > Section`).
    2. **Semantic:** Within those sections, the `chonkie` library performs semantic chunking. It groups sentences by meaning using embedding similarity, falling back to character-limit chunking if semantic boundaries are too ambiguous.
* **Database Infrastructure:** Powered by LanceDB. Features a Hybrid Search setup: combining a vector index (Cosine similarity) for semantic matching with a Full-Text Search (FTS) index for exact keyword hits.
* **Embedding & Reranking Pipeline:**
    * **Embedder:** `nomic-ai/nomic-embed-text-v2-moe` (137M params) - fast CPU-friendly dense retrieval.
    * **Reranker:** Top raw candidates from the database are rescored via cross-encoder using `Alibaba-NLP/gte-reranker-modernbert-base` to ensure only the highest-precision evidence reaches the LLM's context window.

---

## 8. SFT (Supervised Fine-Tuning) Dataset Creation
* **Generative Teachers:** Utilized Llama 3.3 70B and Llama 4 Scout 17B to synthetically generate high-quality adversarial responses.
* **Source Material:** Rooted in real competitive debate cases from OpenCaselist and argument mining data from DebateSum.
* **Dataset Structure:** Configured heavily for ChatML formats. Each row contains: *User Premise*, *Fallacy Analysis*, *RAG Context*, and the *Chosen Response*.
* **Open Source Contribution:** Generated 2,000+ SFT rows and 500 DPO rows. The SFT and adapter datasets are hosted publicly on Hugging Face (`MikeDFT/devils-advocate-sft`) for research reproducibility.

---

## 9. Fine-Tuning the Adversary: SFT Pipeline
* **Infrastructure:** Training executed on Kaggle utilizing the **Unsloth** library, providing significant VRAM savings and up to 2x faster training speeds.
* **Model Architecture:** Started from `unsloth/Qwen3-4B-unsloth-bnb-4bit`.
* **LoRA Configuration:** Applied 4-bit Quantization with QLoRA parameters: Rank = 32, Alpha = 64, targeting projection modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Critical Strategy ("Train-on-Responses-Only"):** Implemented `unsloth.chat_templates.train_on_responses_only`. By masking out the user instructions, RAG context, and system prompts, the loss function is calculated *only* on the assistant's output tokens. This forces the optimizer to focus purely on the adversarial tone and formatting of the rebuttal.
* **Hyperparameters:** Epochs: 2, Sequence Length: 2048, Learning Rate: 2e-4, Batch Size: 4 with Gradient Accumulation of 2.

---

## 10. DPO (Direct Preference Optimization) Dataset
* **The Goal:** Supervised Fine-Tuning teaches the model *how* to debate, but DPO is required to fix underlying personality flaws—specifically, halting sycophancy and reducing RAG hallucinations.
* **Generation:** Generated 2 distinct candidate responses per prompt using the newly fine-tuned SFT adapter.
* **Rigorous Judge Criteria:** Pairs were evaluated by heavily capable models (Llama-3.3-70B and Gemini 2.5 Flash). Responses were scored on adversarial commitment, strict adherence to provided evidence, logical flow, and aggressive tone.
* **Filtering & Penalties:** Applied a strict `MIN_JUDGE_SCORE` of 70. Preference pairs were discarded if both responses were weak. Heavy penalties were applied for agreeing with the user, hedging, apologizing, or inventing facts not in the RAG context.

---

## 11. DPO Training Details
* **Infrastructure:** Executed locally utilizing Unsloth's `PatchDPOTrainer` alongside the `TRL` DPOTrainer. Loaded the SFT model as the 4-bit quantized base.
* **Objective:** Maximize the margin between the chosen (highly adversarial, grounded) and rejected (sycophantic, hallucinated) responses.
* **Hyperparameters:** 
    * Loss Function: Sigmoid
    * Beta (KL Penalty): 0.1
    * Learning Rate: 1e-6 (much lower than SFT to prevent catastrophic forgetting)
    * Sequence Length: 2048
    * Max Steps: 160
    * Precision: BF16/FP16 mixed precision depending on hardware support.

---

## 12. DPO Evaluation & Results
* **Testing Framework:** Evaluated using a custom automated harness (`eval/evaluate.py`) across 50 Benchmark Prompts divided into Easy, Medium, and Hard logical tiers.
* **Overall Score:** Achieved an exceptional **97.4%** success rate across metrics.
    * **100% Use of Evidence:** Zero detected ungrounded claims in the benchmark.
    * **100% Fallacy Detection Integration:** Successfully utilized injected ModernBERT logic in responses.
    * **98% Persona Adherence:** Completely refused to concede or apologize.
    * **98% Hallucination Free:** Reliably rejected claims it lacked evidence for rather than inventing data.
* **Fallacy Classifier Insights:** The standalone ModernBERT classifier yielded 64.7% raw accuracy with 75.7% average confidence on highly complex, out-of-distribution adversarial tests.

---

## 13. System Interface & Conclusion
* **Custom Streamlit GUI:** The agent is packaged within a sleek, custom Streamlit application (`app.py`). It features:
    * Complete multi-chat session management ensuring isolated conversation histories.
    * Expandable debug panels underneath every response showing the exact RAG chunks retrieved.
    * Visual indicators explicitly highlighting triggered logical fallacies.
* **Summary:** The Devil's Advocate successfully proves that small, local models (4B parameters) can be aggressively aligned to combat misinformation and user bias through targeted SFT, preference alignment (DPO), and highly-structured Pipeline RAG.
* **Thank You!** Questions and Live Sparring Demonstrations.
