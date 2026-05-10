import logging
from config import REFORMULATION_TEMPERATURE, REFORMULATION_MAX_TOKENS, REFORMULATION_NUM_QUERIES, LMSTUDIO_UTILITY_MODEL

logger = logging.getLogger(__name__)

class QueryReformulator:
    def __init__(self, client, model_name: str = LMSTUDIO_UTILITY_MODEL):
        self.client = client
        self.model_name = model_name

    def reformulate(self, user_message: str) -> list[str]:
        prompt = (
            f"You are a search query generator for a debate knowledge base.\n"
            f"The user made this argument in a debate:\n"
            f'"{user_message}"\n\n'
            f"Generate exactly {REFORMULATION_NUM_QUERIES} short, specific search queries that would find\n"
            f"EVIDENCE TO COUNTER this argument. Each query should target a different\n"
            f"angle of attack (e.g., counter-evidence, definitional issues, opposing authorities).\n\n"
            f"Return ONLY the queries, one per line, no numbering or bullets."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=REFORMULATION_TEMPERATURE,
                max_tokens=REFORMULATION_MAX_TOKENS,
            )
            content = response.choices[0].message.content.strip()
            # Parse the response by splitting on newlines, stripping empty lines
            queries = [q.strip() for q in content.split("\n") if q.strip()]
            # Return at most REFORMULATION_NUM_QUERIES
            return queries[:REFORMULATION_NUM_QUERIES] if queries else [user_message]
        except Exception as e:
            logger.error(f"Error reformulating query: {e}")
            # Fall back to returning the original input
            return [user_message]
