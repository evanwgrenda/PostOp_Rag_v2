import streamlit as st  # type: ignore
from openai import OpenAI  # type: ignore
from datetime import datetime
from template import PROMPT_TEMPLATE
from langchain.chat_models import ChatOpenAI  # type: ignore
from langchain.chains import ConversationChain  # type: ignore
from langchain.callbacks.tracers import LangChainTracer  # type: ignore
import uuid  # For generating unique run IDs

# Get API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Configure LangChain tracing (requires LANGCHAIN_API_KEY in secrets)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# Instantiate OpenAI client for LangChain
llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=api_key) # type: ignore
conversation_chain = ConversationChain(llm=llm)

# Configure LangChain Tracer
tracer = LangChainTracer()

# Configure Streamlit page
st.set_page_config(
    page_title="UCLA Post-Op Care Assistant",
    layout="centered"  # Optimized for embedded view
)

# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get chatbot response without using a context manager
def get_chatbot_response(user_input):
    try:
        # Generate a unique run ID for tracing
        run_id = str(uuid.uuid4())

        # Log tracing manually if necessary
        tracer.on_chain_start({"name": "conversation_chain"}, {"user_input": user_input}, run_id=run_id) # type: ignore

        # Run the conversation chain
        response = conversation_chain.run(user_input)

        # Log the output manually
        tracer.on_chain_end({"response": response}, run_id=run_id) # type: ignore

        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🏥 UCLA Post-Op Care Assistant")

# Collapsible information section
with st.expander("Important Information"):
    st.warning("For Emergencies Call 911")
    st.info(
        """
        **Urgent Issues (Off hours/weekends)**  
        Call: (310) 206-6766  
        Ask for: Plastic Surgery Resident  

        **Weekday Issues (8AM-5PM)**  
        Call: (310) 794-7616  
        Email: estayton@mednet.ucla.edu  
        """
    )

# Main chat interface
user_input = st.chat_input("Type your question here...")

if user_input:
    with st.spinner('Getting response...'):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get assistant response
        assistant_response = get_chatbot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Display message history in a scrollable container
st.write('<div style="overflow-y: auto; height: 400px;">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).write(content)
st.write('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*This is an AI assistant. For medical emergencies, please call 911 or contact the clinic directly.*")
