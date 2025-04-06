import os
from dotenv import load_dotenv

load_dotenv()

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

import requests

API_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions"
_API_URL = "https://router.huggingface.co/hf-inference/models/meta-llama/Llama-3.3-70B-Instruct/v1/chat/completions" # secondary model for RAG
# OTHER MODELS:
# - https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf
# - https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1
api_key = os.getenv('INFERENCE_API_KEY')
headers = {
    "Accept" : "application/json", 
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "x-use-cache": "false",
    "x-wait-for-model": "true",
}


def query(payload, other_model=False):
    url = _API_URL if other_model else API_URL
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        try:
            return response.json()
        except ValueError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response content: {response.content[:200]}...")
            # return a fallback
            return [{"generated_text": "Sorry, I encountered an error processing your request."}]
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        # return a fallback
        return [{"generated_text": "Sorry, I encountered an error processing your request."}]

class CustomLLM(LLM):
    max_new_tokens: int
    max_time: float

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        other_model: bool = False,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt."""
        # print("####################") 
        # print(prompt)
        # print("####################")

        output = query({
            "inputs": prompt,
            "parameters": {"max_new_tokens": self.max_new_tokens, "max_time": self.max_time, "return_full_text": False},
            "options": {"wait_for_model": True, "use_cache": False}
            }, other_model=other_model)

        output = output[0]["generated_text"]

        if stop is not None:
            for stop_token in stop:
                output = output.split(stop_token)[0]
            
        return output
    
    def call_chat_completion(
        self,
        messages: List[dict],
        other_model: bool = False,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Run the LLM using chat completion format."""

        response = query({
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "model": "meta-llama/Llama-3.3-70B-Instruct"
        }, other_model=other_model)

        output = response["choices"][0]["message"]["content"]

        if stop is not None:
            for stop_token in stop:
                output = output.split(stop_token)[0]
            
        return output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_new_tokens": self.max_new_tokens, "max_time": self.max_time}


if __name__ == "__main__":
    # create a new LLM instance
    llm = CustomLLM(max_new_tokens=250, max_time=60.0)
    # run the LLM
    print(llm("Can you please let us know more details about your "))
