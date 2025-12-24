import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import aiohttp

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekClient:
    """Production-grade DeepSeek client with retry logic and monitoring."""

    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com"
        self.model = settings.DEEPSEEK_MODEL
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = aiohttp.ClientTimeout(total=30)

        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "failed_requests": 0,
            "last_request": None
        }

        # Session management
        self.session = None

    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )

    async def chat_completion(self,
                              messages: List[Dict[str, str]],
                              temperature: float = 0.1,
                              max_tokens: int = 1000,
                              stream: bool = False,
                              **kwargs) -> str:
        """Execute chat completion with retry logic."""
        await self._ensure_session()
        self.usage_stats["total_requests"] += 1

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        # Add any additional parameters
        payload.update(kwargs)

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"DeepSeek request (attempt {attempt + 1}): {len(messages)} messages")

                async with self.session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload
                ) as response:

                    if response.status == 200:
                        data = await response.json()

                        # Update usage stats
                        if "usage" in data:
                            self.usage_stats["total_tokens"] += data["usage"]["total_tokens"]

                        self.usage_stats["last_request"] = datetime.now()

                        return data["choices"][0]["message"]["content"]

                    elif response.status == 429:  # Rate limit
                        logger.warning(f"DeepSeek rate limit exceeded (attempt {attempt + 1})")
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (2 ** attempt)
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            await asyncio.sleep(wait_time)
                            continue

                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error {response.status}: {error_text}")

                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                            continue

            except aiohttp.ClientError as e:
                logger.error(f"DeepSeek connection error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.usage_stats["failed_requests"] += 1
                    raise

            except asyncio.TimeoutError:
                logger.error(f"DeepSeek timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * 2)
                else:
                    self.usage_stats["failed_requests"] += 1
                    raise

            except Exception as e:
                logger.error(f"Unexpected DeepSeek error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    self.usage_stats["failed_requests"] += 1
                    raise

        # Max retries exceeded
        self.usage_stats["failed_requests"] += 1
        raise Exception("DeepSeek API request failed after max retries")

    async def chat_completion_with_tools(self,
                                         messages: List[Dict[str, str]],
                                         tools: List[Dict[str, Any]],
                                         tool_choice: Optional[Union[str, Dict]] = "auto",
                                         **kwargs) -> Dict[str, Any]:
        """Execute chat completion with tool calling."""
        await self._ensure_session()

        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            **kwargs
        }

        try:
            async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
            ) as response:

                if response.status == 200:
                    data = await response.json()

                    # Update usage
                    if "usage" in data:
                        self.usage_stats["total_tokens"] += data["usage"]["total_tokens"]

                    message = data["choices"][0]["message"]

                    result = {
                        "content": message.get("content"),
                        "role": message.get("role"),
                        "finish_reason": data["choices"][0].get("finish_reason")
                    }

                    if "tool_calls" in message and message["tool_calls"]:
                        tool_calls = []
                        for tool_call in message["tool_calls"]:
                            tool_calls.append({
                                "id": tool_call.get("id"),
                                "type": tool_call.get("type"),
                                "function": {
                                    "name": tool_call["function"].get("name"),
                                    "arguments": json.loads(tool_call["function"].get("arguments", "{}"))
                                }
                            })
                        result["tool_calls"] = tool_calls

                    return result

                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek tools API error {response.status}: {error_text}")
                    raise Exception(f"DeepSeek API error: {response.status}")

        except Exception as e:
            logger.error(f"DeepSeek tool calling failed: {e}")
            raise

    async def embeddings(self,
                         input_text: Union[str, List[str]],
                         model: str = "deepseek-embed") -> List[List[float]]:
        """Get embeddings from DeepSeek."""
        await self._ensure_session()

        payload = {
            "model": model,
            "input": input_text if isinstance(input_text, list) else [input_text]
        }

        try:
            async with self.session.post(
                    f"{self.base_url}/embeddings",
                    json=payload
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    return [item["embedding"] for item in data["data"]]

                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek embeddings error {response.status}: {error_text}")
                    raise Exception(f"DeepSeek embeddings error: {response.status}")

        except Exception as e:
            logger.error(f"DeepSeek embeddings failed: {e}")
            raise

    async def stream_chat_completion(self,
                                     messages: List[Dict[str, str]],
                                     temperature: float = 0.1,
                                     max_tokens: int = 1000,
                                     **kwargs) -> Any:
        """Stream chat completion response."""
        await self._ensure_session()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs
        }

        try:
            async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
            ) as response:

                if response.status == 200:
                    async for chunk in response.content:
                        if chunk:
                            chunk_str = chunk.decode('utf-8').strip()
                            if chunk_str.startswith('data: '):
                                data_str = chunk_str[6:]
                                if data_str != '[DONE]':
                                    try:
                                        data = json.loads(data_str)
                                        yield data
                                    except json.JSONDecodeError:
                                        pass

                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek stream error {response.status}: {error_text}")
                    raise Exception(f"DeepSeek stream error: {response.status}")

        except Exception as e:
            logger.error(f"DeepSeek stream failed: {e}")
            raise

    async def moderate(self, text: str) -> Dict[str, Any]:
        """Moderate text using DeepSeek."""
        # DeepSeek doesn't have a dedicated moderation API, so we use chat completion
        messages = [
            {
                "role": "system",
                "content": "Analyze the following text for harmful content. "
                           "Return a JSON with 'flagged' (boolean) and 'categories' (list of strings)."
            },
            {
                "role": "user",
                "content": text
            }
        ]

        try:
            response = await self.chat_completion(
                messages=messages,
                temperature=0,
                max_tokens=100
            )

            # Parse response
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                # Fallback to simple analysis
                harmful_keywords = ["harmful", "dangerous", "violent", "hate", "abuse"]
                flagged = any(keyword in text.lower() for keyword in harmful_keywords)

                return {
                    "flagged": flagged,
                    "categories": ["manual_check"] if flagged else []
                }

        except Exception as e:
            logger.error(f"DeepSeek moderation failed: {e}")
            return {"flagged": False, "error": str(e)}

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
        """Validate DeepSeek API key."""
        try:
            # Try a simple completion to validate key
            messages = [{"role": "user", "content": "Hello"}]
            await self.chat_completion(messages, max_tokens=5)
            logger.info("DeepSeek API key validated successfully")
            return True
        except Exception as e:
            logger.error(f"DeepSeek API key validation failed: {e}")
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        await self._ensure_session()

        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    return {"error": f"API returned {response.status}"}
        except Exception as e:
            logger.error(f"Failed to get DeepSeek models: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("DeepSeek client session closed")


# Global instance
deepseek_client = DeepSeekClient()


async def get_deepseek_client() -> DeepSeekClient:
    """Get DeepSeek client instance."""
    return deepseek_client


async def close_deepseek_client():
    """Close DeepSeek client."""
    await deepseek_client.close()