from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random


class LLM:
    def __init__(self, model_name, openai_api_key, max_tokens=None):
        self.llm = ChatOpenAI(temperature=0, model_name=model_name,
                              openai_api_key=openai_api_key, max_tokens=max_tokens)

    def simple_generate(self, prompt):
        messages_prompt = [SystemMessage(content=prompt)]
        llm_response = self.llm.generate([messages_prompt])
        return llm_response.generations[0][0].text

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(55) + wait_random(0, 60))
    async def async_simple_generate(self, prompt):
        messages_prompt = [SystemMessage(content=prompt)]
        llm_response = await self.llm.agenerate([messages_prompt])
        return llm_response.generations[0][0].text
