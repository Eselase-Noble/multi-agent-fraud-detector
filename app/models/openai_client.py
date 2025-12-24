import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import openai
from openai import AsyncOpenAI, RateLimitError, APIError

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Production-grade OpenAI client with retry logic and monitoring."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.max_retries = 3
        self.retry_delay = 1.0

        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "failed_requests": 0,
            "last_request": None
        }

        # Rate limiting
        self.rate_limit = {
            "requests_per_minute": 60,
            "tokens_per_minute": 150000,
            "last_reset": datetime.now()
        }

    async def chat_completion(self,
                              messages: List[Dict[str, str]],
                              temperature: float = 0.1,
                              max_tokens: int = 1000,
                              stream: bool = False,
                              **kwargs) -> Union[str, Any]:
        """Execute chat completion with retry logic."""
        self.usage_stats["total_requests"] += 1

        for attempt in range(self.max_retries):
            try:
                # Check rate limits
                await self._check_rate_limit()

                logger.debug(f"OpenAI request (attempt {attempt + 1}): {len(messages)} messages")

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    **kwargs
                )

                # Update usage stats
                if hasattr(response, 'usage'):
                    self.usage_stats["total_tokens"] += response.usage.total_tokens

                self.usage_stats["last_request"] = datetime.now()

                if stream:
                    return response
                else:
                    return response.choices[0].message.content

            except RateLimitError as e:
                logger.warning(f"OpenAI rate limit exceeded (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    self.usage_stats["failed_requests"] += 1
                    raise

            except APIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.usage_stats["failed_requests"] += 1
                    raise

            except Exception as e:
                logger.error(f"Unexpected error in OpenAI request (attempt {attempt + 1}): {e}")
                self.usage_stats["failed_requests"] += 1
                raise

        # Should never reach here
        raise Exception("Max retries exceeded")

    async def chat_completion_with_functions(self,
                                             messages: List[Dict[str, str]],
                                             functions: List[Dict[str, Any]],
                                             function_call: Optional[Union[str, Dict]] = "auto",
                                             **kwargs) -> Dict[str, Any]:
        """Execute chat completion with function calling."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=functions,
                function_call=function_call,
                **kwargs
            )

            message = response.choices[0].message

            result = {
                "content": message.content,
                "role": message.role,
                "finish_reason": response.choices[0].finish_reason
            }

            if message.function_call:
                result["function_call"] = {
                    "name": message.function_call.name,
                    "arguments": json.loads(message.function_call.arguments)
                }

            # Update usage
            if hasattr(response, 'usage'):
                self.usage_stats["total_tokens"] += response.usage.total_tokens

            return result

        except Exception as e:
            logger.error(f"OpenAI function calling failed: {e}")
            raise

    async def embeddings(self,
                         input_text: Union[str, List[str]],
                         model: str = "text-embedding-3-small") -> List[List[float]]:
        """Get embeddings for text."""
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=input_text if isinstance(input_text, list) else [input_text],
                encoding_format="float"
            )

            embeddings = [data.embedding for data in response.data]

            # Update usage
            if hasattr(response, 'usage'):
                self.usage_stats["total_tokens"] += response.usage.total_tokens

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embeddings failed: {e}")
            raise

    async def moderate(self, text: str) -> Dict[str, Any]:
        """Moderate text for safety."""
        try:
            response = await self.client.moderations.create(input=text)

            result = {
                "flagged": response.results[0].flagged,
                "categories": response.results[0].categories,
                "category_scores": response.results[0].category_scores
            }

            return result

        except Exception as e:
            logger.error(f"OpenAI moderation failed: {e}")
            return {"flagged": False, "error": str(e)}

    async def _check_rate_limit(self):
        """Check and enforce rate limits."""
        now = datetime.now()

        # Reset counter if minute has passed
        if (now - self.rate_limit["last_reset"]).seconds >= 60:
            self.rate_limit["last_reset"] = now
            # In production, would reset counters here

        # For now, just log (production would have proper rate limiting)
        if self.usage_stats["total_requests"] % 10 == 0:
            logger.debug(f"OpenAI usage: {self.usage_stats['total_requests']} requests, "
                         f"{self.usage_stats['total_tokens']} tokens")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.usage_stats,
            "success_rate": (
                (self.usage_stats["total_requests"] - self.usage_stats["failed_requests"]) /
                self.usage_stats["total_requests"] * 100
                if self.usage_stats["total_requests"] > 0 else 0
            ),
            "model": self.model,
            "avg_tokens_per_request": (
                self.usage_stats["total_tokens"] / self.usage_stats["total_requests"]
                if self.usage_stats["total_requests"] > 0 else 0
            )
        }

    async def validate_api_key(self) -> bool:
        """Validate OpenAI API key."""
        try:
            await self.client.models.list()
            logger.info("OpenAI API key validated successfully")
            return True
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {e}")
            return False

    async def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Failed to get OpenAI models: {e}")
            return []

    async def create_fine_tune_job(self,
                                   training_file: str,
                                   validation_file: Optional[str] = None,
                                   model: Optional[str] = None) -> Dict[str, Any]:
        """Create a fine-tuning job."""
        try:
            response = await self.client.fine_tuning.jobs.create(
                training_file=training_file,
                validation_file=validation_file,
                model=model or self.model,
                hyperparameters={
                    "n_epochs": 3,
                    "batch_size": 4,
                    "learning_rate_multiplier": 0.1
                }
            )

            return {
                "job_id": response.id,
                "status": response.status,
                "model": response.model,
                "created_at": response.created_at
            }

        except Exception as e:
            logger.error(f"Failed to create fine-tuning job: {e}")
            raise


# Global instance
openai_client = OpenAIClient()


async def get_openai_client() -> OpenAIClient:
    """Get OpenAI client instance."""
    return openai_client