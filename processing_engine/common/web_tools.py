import asyncio
import random
from ddgs import DDGS


def search_web(query: str) -> str:
    """Search the web for the query using DuckDuckGo (ddgs). Returns the results as a string."""
    results = DDGS().text(query, max_results=5, region="in-en")
    return str(results)


async def search_web_async(query: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, search_web, query)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
                continue
            else:
                return f"Web search verification failed after {max_retries} attempts: {str(e)}"
    return f"Web search verification failed after {max_retries} attempts."


def web_verify_medicine(medicine_name: str) -> str:
    query = f"{medicine_name} drug medication pharmaceutical"
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(search_web_async(query))
        return result
    except RuntimeError:
        return asyncio.run(search_web_async(query))
