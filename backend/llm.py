import os
from dotenv import load_dotenv

load_dotenv()

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

import requests

# API_URL = "https://s1bc92t3401pk41g.us-east-1.aws.endpoints.huggingface.cloud" # primary model for chat
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"
_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf" # secondary model for RAG
# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
api_key = os.getenv('INFERENCE_API_KEY')
headers = {
    "Accept" : "application/json", 
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json" 
}


def query(payload, other_model=False):
	response = requests.post(API_URL, headers=headers, json=payload) if not other_model else requests.post(_API_URL, headers=headers, json=payload)
	return response.json()

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

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_new_tokens": self.max_new_tokens, "max_time": self.max_time}


if __name__ == "__main__":
    # create a new LLM instance
    llm = CustomLLM(max_new_tokens=250, max_time=60.0)
    # run the LLM
    print(llm("Can you please let us know more details about your "))
