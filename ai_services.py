"""
AI Services 
OpenRouter Integration 

Handles:
1. Profile Sanitization + Enrichment
2. Embedding Generation
3. AI Introduction with Safety Moderation
"""
from dotenv import load_dotenv
import os
load_dotenv()
import logging

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
logger = logging.getLogger(__name__)

from openai import OpenAI
import os
import json
import re
from typing import List, Dict, Tuple


class AIService:
    def __init__(self):

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            logger.warning("OPENROUTER_API_KEY not set — AI features disabled")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Community-Matching-System"
                }
            )

        self.embedding_model = "openai/text-embedding-3-small"
        self.chat_model = "openai/gpt-4o-mini"
        self.moderation_model = "openai/omni-moderation-latest"


    def sanitize_and_enrich_profile(
        self,
        bio: str,
        interest_tags: List[str]
    ) -> Tuple[str, List[str], bool]:

        prompt = f"""You are a profile sanitization assistant.

1. Remove any PII (phone numbers, emails, addresses)
2. Clean the bio
3. Extract implied interest tags
4. Return enriched tags

Bio: "{bio}"
Current Tags: {interest_tags}

Respond ONLY in JSON:
{{
    "sanitized_bio": "text",
    "enriched_tags": ["tag1", "tag2"],
    "pii_found": true/false
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a data sanitization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            content = response.choices[0].message.content.strip()
            result = json.loads(content)

            return (
                result["sanitized_bio"],
                result["enriched_tags"],
                result["pii_found"]
            )

        except Exception:
            # Fallback if model fails
            cleaned = self._basic_pii_removal(bio)
            return cleaned, interest_tags, False

    def _basic_pii_removal(self, text: str) -> str:
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
                      '[email removed]', text)

        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                      '[phone removed]', text)

        return text

    def generate_embedding(self, text: str) -> List[float]:

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )

        return response.data[0].embedding

    def create_embedding_payload(self, bio: str, tags: List[str]) -> str:
        tags_text = ", ".join(tags)
        return f"Bio: {bio}\nInterests: {tags_text}"


    def generate_ai_introduction(
        self,
        user_bio: str,
        community_name: str,
        community_description: str,
        active_members: List[Dict]
    ) -> Tuple[str, str, float]:

        mentioned_member = (
            active_members[0]["username"]
            if active_members else None
        )

        prompt = f"""Generate a warm 2-3 sentence welcome message.

Community: {community_name}
Description: {community_description}
New Member Bio: {user_bio}
Mention: @{mentioned_member}

Only return the message text.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a community onboarding assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            intro_text = response.choices[0].message.content.strip()

            # Run moderation check
            toxicity_score = self._check_toxicity(intro_text)

            return intro_text, mentioned_member, toxicity_score

        except Exception:
            return (
                f"Welcome to {community_name}! We're excited to have you here.",
                None,
                0.0
            )


    def _check_toxicity(self, text: str) -> float:
        """
        Returns highest moderation score (0.0 - 1.0)
        """

        try:
            response = self.client.moderations.create(
                model=self.moderation_model,
                input=text
            )

            scores = response.results[0].category_scores
            return max(
                scores.hate,
                scores.harassment,
                scores.violence,
                scores.sexual
            )

        except Exception:
            return 0.0


# Global instance
ai_service = AIService()
