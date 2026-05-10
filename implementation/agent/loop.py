"""
Interactive Devil's Advocate agent (Pipeline RAG).

Implements a Pipeline RAG loop using LMStudio's OpenAI-compatible API.
Python automatically fetches the RAG context and Fallacy detection before
sending the strictly formatted prompt to the fine-tuned model.

Requires LMStudio running with the fine-tuned GGUF model loaded.
Start with: lms server start
"""

import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, CONTEXT_WINDOW_EXCHANGES, LMSTUDIO_MAIN_MODEL, LMSTUDIO_UTILITY_MODEL, SYSTEM_PROMPT
from agent.tools import ToolExecutor
from agent.reformulator import QueryReformulator


class DevilsAdvocate:
    def __init__(self, model_name=None):
        self.client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY)
        self.tool_executor = ToolExecutor(llm_client=self.client, utility_model=LMSTUDIO_UTILITY_MODEL)
        self.reformulator = QueryReformulator(self.client, LMSTUDIO_UTILITY_MODEL)
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.model_name = model_name

    def _get_model_name(self):
        """Auto-detect the loaded model if not specified."""
        if self.model_name:
            return self.model_name
        try:
            models = self.client.models.list()
            if models.data:
                self.model_name = models.data[0].id
                return self.model_name
        except Exception:
            pass
        return LMSTUDIO_MAIN_MODEL

    def respond(self, user_message: str) -> dict:
        """Process a user message through the Pipeline RAG flow.

        Returns a dict with:
            reply                (str)  - the agent's response
            reformulated_queries (list) - queries generated for RAG retrieval
            rag_context          (str)  - retrieved and reranked context chunks
            fallacy_context      (str)  - fallacy detection result + explanation
        """
        # Reformulate queries and fetch multi-RAG context
        queries = self.reformulator.reformulate(user_message)
        rag_context = self.tool_executor.execute_multi_rag(queries, user_message)

        # Fallacy detection
        fallacy_args = json.dumps({"argument_text": user_message})
        fallacy_context = self.tool_executor.execute("check_fallacy", fallacy_args)

        # Format the payload

        # formatted_prompt = (
        #     f"Context Information:\n{rag_context}\n\n"
        #     f"Fallacy Analysis:\n{fallacy_context}\n\n"
        #     f"User Argument:\n{user_message}"
        # )

        formatted_prompt = (
            "The user made the following debate claim:\n\n"
            f"{user_message}\n\n"
            "Fallacy detector result:\n"
            f"{fallacy_context}\n\n"
            "Retrieved evidence:\n"
            f"{rag_context}\n\n"
            "Write the Devil's Advocate response. Attack the user's claim directly. "
            "Use only the retrieved evidence for factual claims. "
            "If the fallacy detector says no fallacy, do not invent one. "
            "Do not agree with the user."
        )

        # Append the formatted string to history and trim if needed
        self.history.append({"role": "user", "content": formatted_prompt})
        self._trim_history()

        # Generate the response
        response = self.client.chat.completions.create(
            model=self._get_model_name(),
            messages=self.history,
            temperature=0.7,
        )

        reply = response.choices[0].message.content or ""

        # Save the assistant's reply to history
        self.history.append({"role": "assistant", "content": reply})

        return {
            "reply": reply,
            "reformulated_queries": queries,
            "rag_context": rag_context,
            "fallacy_context": fallacy_context,
        }

    def _trim_history(self):
        """Keep conversation within context limits by trimming old exchanges."""
        exchanges = sum(1 for m in self.history if m["role"] == "user")
        if exchanges <= CONTEXT_WINDOW_EXCHANGES:
            return

        system = self.history[0]
        keep_from = len(self.history)
        kept = 0
        for i in range(len(self.history) - 1, 0, -1):
            if self.history[i]["role"] == "user":
                kept += 1
                if kept >= CONTEXT_WINDOW_EXCHANGES:
                    keep_from = i
                    break

        trimmed_count = keep_from - 1
        self.history = [system] + self.history[keep_from:]
        if trimmed_count > 0:
            print(f"  [Trimmed {trimmed_count} old messages from context]")

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("Conversation reset.")


def main():
    print("=" * 60)
    print("  THE DEVIL'S ADVOCATE")
    print("  An adversarial debate sparring partner (Pipeline RAG)")
    print("=" * 60)
    print("\nConnecting to LMStudio at", LMSTUDIO_BASE_URL)
    print("Commands: 'quit', 'reset', 'history'\n")

    agent = DevilsAdvocate()

    try:
        model = agent._get_model_name()
        print(f"Model loaded: {model}\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to LMStudio: {e}")
        print("Make sure LMStudio is running with 'lms server start'")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue
        if user_input.lower() == "history":
            for msg in agent.history:
                role = msg["role"].upper()
                content = msg.get("content", "")
                if content:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"  [{role}] {preview}")
            continue

        try:
            result = agent.respond(user_input)
            print(f"\nDevil's Advocate: {result['reply']}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
