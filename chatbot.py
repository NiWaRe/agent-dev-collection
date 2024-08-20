import weave
from openai import OpenAI
import streamlit as st
from uuid import uuid4
import requests
import asyncio

MODEL_WEAVE_URL = "weave:///wandb-smle/weave-rag-experiments/object/RagModel:cqRaGKcxutBWXyM0fCGTR1Yk2mISLsNari4wlGTwERo"
MODEL_LOCAL_URL = "http://localhost:9996/predict"

# Generic Streamlit Chatbot Code

st.title("Sample Chatbot with User Feedback")

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Specific Code Involving Weave
# TODO; why no @st.cache_resource?
def init_weave():
    client = weave.init('wandb-smle/weave-cookboook-demo') #'wandb-smle/weave-rag-experiments')

@st.cache_resource
def get_weave_model(weave_url: str):
    model = weave.ref(weave_url).get()
    return model

# TODO: this weave op should not be necessary
# TODO: the comment feature doesn't really work
@weave.op()
def production_chat_response(model_url: str, prompt: str):
    """
    The main model function that will take the prompt and the model_url to generate a response.
    Optionally, this function call also be decorated with @weave.op(). 
    """
    
    if model_url.startswith("weave://"):
        # option 1 - get the model object from weave
        RagModel = get_weave_model(model_url)
        
        # NOTE: we use the .call object so that the call is returned and feedback can be added
        # TODO: check answer from Adam if i still need to a the specific model object in the call
        #data, call = asyncio.run(RagModel.predict.call(query=prompt, n_documents=2))  
        data = asyncio.run(RagModel.predict(prompt))   
        result_content = data['result']['content']
        call = weave.get_current_call()  

    elif model_url.startswith("http"):
        # option 2 - the model is already served to a certain endpoint
        response = requests.post(
            model_url,
            json={
                "query": prompt,
                "n_documents": 2,
            },
        )
        data = response.json()
        result_content = data['result']['result']['content']

        # TODO: this works when this is a weave.op - how to get the last call otherwise?
        call = weave.get_current_call() 
    else:
        print("Please pass in a URL to a weave object or http endpoint.")
    
    
    # Display the main response content
    st.markdown(result_content)
    
    return result_content, call

def get_and_process_prompt(model_url: str):
    # capture input
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # show user input
        with st.chat_message("user"):
            st.markdown(prompt)

        # genereate response ad visualize assistant answer
        with st.chat_message("assistant"):
            with weave.attributes({'session': st.session_state['session_id']}):
                response, call = production_chat_response(model_url, prompt)
                
                # add general text feedback field for users to give text feedback
                st.markdown("Your feedback: ")
                feedback_text = st.text_input("", placeholder="Type your thoughts...")
                st.button(":thumbsup:",   on_click=lambda: call.feedback.add_reaction("👍"), key='up')
                st.button(":thumbsdown:", on_click=lambda: call.feedback.add_reaction("👎"), key='down')
                st.button("Submit Feedback", on_click=lambda: call.feedback.add_note(feedback_text), key='submit')
                
                # save assistant response in general flow of application
                st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    session_id = str(uuid4())
    st.session_state['session_id'] = session_id
    init_weave()
    init_chat_history()
    display_chat_messages()
    get_and_process_prompt(model_url = MODEL_WEAVE_URL)

if __name__ == "__main__":
    main()
