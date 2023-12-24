import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS

from backend.core import run_llm, get_faiss_vectordb


##################################################################################
# Code refactored from https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/ #
##################################################################################

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "I'm your course buddy. How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "I'm your course buddy. How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Format for Llama2 chatbot following the LLaMA2 template
# <s>[INST] <<SYS>>
# System prompt
# <</SYS>>
#
# User prompt [/INST] Model answer </s>
def chat_completion(prompt):
    string_dialogue = "<s>[INST] <<SYS>>\n"
    string_dialogue += prompt
    string_dialogue += "\n<</SYS>>\n\n"

    # skip the first message since it is the system prompt
    for dict_message, i in zip(st.session_state.messages[1:], range(len(st.session_state.messages[1:]))):
        if dict_message["role"] == "user":
            string_dialogue += dict_message["content"] if i == 0 else "[INST] " + dict_message["content"]
            string_dialogue += " [/INST] "
        else:
            string_dialogue += dict_message["content"] + " "
    # string_dialogue += "</s>"

    return string_dialogue

# Function for generating RAG response
def run_rag(prompt_input, search):
    with st.spinner("Retrieving results..."):

        db = get_faiss_vectordb(inference_api_key=os.getenv('INFERENCE_API_KEY'))

        # load the FAISS vector database from the generated index path
        # retrieve the top 5 most similar documents
        docs_and_scores = db.similarity_search_with_score(search)[:5]

        # TODO: The answer is cut off after a while. Fix this. The reasone is the llm returns text with \n characters and bulletpoints, which occupy space
        prompt = "<s>[INST] <<SYS>>\nAnswer a question using References. Remember to mention the source(s) you use. For example, if 'Source: 01_introduction.pdf, Page: 8', you can say 'According to page 8 in slides 01_introduction.pdf, ...'.\n<</SYS>>\n\n"
        prompt += f"Question: {prompt_input} References: "
        for doc, score in docs_and_scores:
            prompt += f"{doc} "
        prompt += "[/INST] "

        return run_llm(prompt, stop=["</s>"]).strip()

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    prompt = "Assist a student who is taking a university course. When you respond, you may respond on the knowledge you have or you may perform an Action, ONLY if necessary, i.e., the student asks for information about the course material. Action can be of one type only: (1) Search[content], which searches for similar content in the course material and returns the most relevant results if they exist. Put what you want to search for inside brackets after Search, like this: Search[What is the exam like?]. Be sure to put something that is inherent to the student's question."
    # examples = "\n<s>[INST] Hi, Im Bob! [/INST] Hi Bob, how can I help you today? [INST] I need information about the exam. Can you help me? [/INST] Sure, what do you want to know about the exam? [INST] How is the exam composed? [/INST] Search[How is the exam composed?]</s>"
    # prompt += examples

    string_dialogue = chat_completion(prompt)
    
    query = f"{string_dialogue}"
    output = run_llm(query=query, stop=["</s>"]).strip()

    # extract the Text if the Search[Text] action is performed
    if "Search[" in output:
        search = output.split("Search[")[1].split("]")[0]
        results = run_rag(prompt_input, search)
        output = "Search[" + search + "]\n\n" + results

    return output

# User-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    print("\n\nsession_state.messages: ", st.session_state.messages, "\n\n")

# TODO: Regenerate response if user edits prompt or unsatisfied with response
