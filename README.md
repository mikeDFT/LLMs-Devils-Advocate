## Project name:
The Devil's Advocate - Agentic SLM debate sparring partner

## Team Name
Source: Trust Me

## Team Members
- Ban Mihai-Constantin
- Barbos Andrada-Ioana
- Beudean Carmen-Laura
- Alexe Nicolae-Razvan

## Introduction
A debate opponent is running locally. You give it a thesis, it argues against you using real retrieved evidence, catches logical fallacies in your reasoning, and won't concede until you beat it. Built on an SLM trained for debating through fine-tuning, RAG grounding and preference alignment.

## Model Qwen3-4B
Chosen over Llama 3.2 3B and Phi-4-mini:
The biggest reason is that Qwen3 has a built-in thinking mode you can toggle on and off (there are Qwen models that restrict only one mode though). When it's reasoning through an argument, the model silently works through a chain of thought before it responds. When it's calling tools, thinking gets turned off for speed. The second reason is tool calling. Qwen3-4B is one of the best sub-8B models at structured function calling and it knows when to call a tool and when not to, which sounds obvious but is a real problem with smaller models. It also fits in about 10GB VRAM, as that's the most we can do. For inference it runs as a Q4_K_M GGUF quantization (basically compressed) through LMStudio, using only about 3GB of VRAM. Training uses 4-bit QLoRA through Unsloth, which keeps it under 10GB VRAM.

## Training datasets
A few datasets were chosen for supervised fine-tuning, each teaching a different aspect of debate. We may not use all of them if training takes too long or we deem them unnecessary, but we found:
OpenCaselist (https://huggingface.co/datasets/Yusuf5/OpenCaselist) is the largest one, with about 4.95 million rows of real competitive debate cases.
DebateSum (https://huggingface.co/datasets/Hellisotherpeople/DebateSum) is a large-scale argument mining and summarization dataset with 241k rows.
pie/aae2 (https://huggingface.co/datasets/pie/aae2) is a collection of 402 essays with over 6.000 annotated argument components.
All of this data should get reformatted into Qwen3's chat template before training.

## Fine-tuning
Fine-tuning uses QLoRA (Quantized Low-Rank Adaptation) through the Unsloth library, which makes training possible on our hardware. We need to try out some parameter values to see what will work out best here.

## RAG pipeline
The RAG pipeline should stop the model from making up evidence (aka hallucinate). Every argument it makes gets grounded in real retrieved sources before the model generates its response.
Knowledge sources
Stanford Encyclopedia of Philosophy (SEP) covers epistemology, ethics, formal logic, and philosophy of mind.
NSDA Unified Manuals contain the official rules for competitive debate such as evidence standards, burdens of proof, what counts as a valid argument.
## Chunking strategy
The chunking library we use is chonkie, which handles two levels of splitting.
The first pass is splitting around headers, for structured sources like SEP, the document gets broken up at each header before anything else, so chunks don't accidentally cross section boundaries. Each chunk keeps its full header chain as metadata, so the retrieval system always knows the context (for example: SEP > Ethics > Utilitarianism > Act vs Rule).
The second pass is semantic chunking, within each header section, splits happen at meaning boundaries using embedding similarity rather than arbitrary token counts. This keeps related sentences together instead of cutting them off mid-thought.
Before any chunk gets indexed, the model prepends a 1-2 sentence summary describing where in the document the chunk comes from. This gives the embedding model more context during retrieval and should improve precision on ambiguous queries.
## Vector database
The vector database will most likely be LanceDB (we're also considering ChromaDB).
## Embedding model
The embedding model is nomic-ai/nomic-embed-text-v2-moe, a 137-million parameter model. It's widely used and runs on CPU fast enough for real-time retrieval.
## Retrieval strategy
Each query goes through: hybrid search retrieves the top 20 candidates using both BM25 keyword matching (exact keyword matching) and dense vector similarity. Cross-encoder reranking with mxbai-rerank-base-v2 (143M parameters) rescores all 20 candidates by looking at the full (query, document) pair together, rather than scoring them independently. This is much more accurate but too slow to run on all documents, that's why the initial retrieval narrows it down first. The top 5 reranked documents get injected into the model's context as grounding evidence.

## Fallacy detection
The fallacy detector is a separate system that runs every time the user submits an argument.
Firstly, a fast classification pass using a fine-tuned ModernBERT-base (150M parameters). ModernBERT is a 2024 architecture that replaced DeBERTa, it's twice as fast, supports up to 8,192 tokens of context (compared to DeBERTa's 512), and uses Flash Attention. It gets fine-tuned to detect 13 fallacy types including ad hominem, straw man, false causality, circular reasoning, and appeal to emotion.
Two datasets train the classifier: tasksource/logical-fallacy with 3.761 labeled examples across those 13 types, and causalNLP/propaganda-detection with around 20.000 examples covering 18 propaganda and persuasion techniques.
Then if a fallacy is detected, the classification result gets passed to Qwen3-4B, which generates a natural language explanation of the specific fallacy and creates a counter argument targeting the user's actual position, not a misrepresentation of it.
The whole classification shouldn't add noticeable latency, but we need to test this.

## Preference alignment, DPO
After fine-tuning, the model gets a second round of training specifically to fix its personality problems, mainly agreeing when it shouldn't and hallucinations.
The method we're using is Direct Preference Optimization (DPO) with sigmoid loss, run through TRL's DPOTrainer. The original plan was SimPO, which is theoretically better because it doesn't need a reference model (saving VRAM), but we had some technical issues with the libraries. Standard DPO still fits in 12GB VRAM because Unsloth reuses the base model weights implicitly instead of loading a separate copy.
## Preference datasets
argilla/ultrafeedback-binarized-preferences-cleaned (https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) has about 60.000 preference pairs, professionally cleaned to remove ambiguous examples. This is the main dataset and covers general response quality.
tasksource/logical-fallacy (https://huggingface.co/datasets/tasksource/logical-fallacy) has around 2.680 rows.
jondurbin/truthy-dpo-v0.1 (https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) has 1.000 pairs specifically targeting hallucination and sycophancy. It's small but directly relevant to what we're trying to fix.
Synthetic debate pairs are generated from the fine-tuned model itself. About 500 debate prompts are taken from OpenCaselist, two responses are generated per prompt, and a strong judge model scores them against a debate rubric. The chosen response is the one that's factually grounded and attacks structural weaknesses; the rejected response is the one that hallucinates evidence, breaks character, or gives in to the user. This is the most valuable dataset for our specific goal.
The most important thing about the DPO data is that every chosen response has to embody the Devil's Advocate persona. If any of the chosen responses are polite or accommodating, the alignment stage will overwrite the adversarial behavior learned during fine-tuning.

## Agentic behaviour
The whole system runs through a custom Python loop implementing the ReAct (Reason + Act) pattern.
When the user submits a query, the loop starts. Qwen3-4B processes the conversation history with thinking mode on, silently works through its reasoning, then decides what to do. It calls the RAG tool, which queries LanceDB and returns the top 5 reranked documents. To check for fallacies it calls the fallacy tool, which runs ModernBERT and optionally generates a critique. Once it has everything it needs, it generates a plain text response and the loop waits for the next user message.
The model is served locally through LMStudio, the one we used during the laboratory.
