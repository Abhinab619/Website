from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

app = FastAPI()

# Allow frontend requests (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model to receive messages from frontend
class Message(BaseModel):
    text: str

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Vector Store & Retriever
vectorstore = Chroma(
persist_directory = os.path.join(BASE_DIR, "data_store"),
    embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key='AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE')
)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_store_path = os.path.join(BASE_DIR, "data_store")

print("Absolute Path of data_store:", data_store_path)


retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.7})

retriever_tool = create_retriever_tool(
    retriever=retriever, name="Udhami_Yojna", description="Retrieves relevant information from stored documents sumarizing all the inforamtion without missing any"
)

# Direct Gemini Tool
chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                                  google_api_key='AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE'
)

@tool
def direct_llm_answer(query: str) -> str:
    """Directly generates an answer from the LLM."""
    response = chat.invoke(query)
    return response


# Adding a memory for context Retention

from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=chat, memory_key="chat_history",return_messages=True)



# Define tools
tools = [retriever_tool, direct_llm_answer]

chat_prompt_template = hub.pull("hwchase17/openai-tools-agent")

# Create tool calling agent
agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=chat_prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, memory = memory)

interaction_count = 0
MAX_MEMORY_SIZE = 5  # Clear memory after 10 interactions

@app.post("/chat")
def chat_with_model(msg: Message):
    global interaction_count
    
    # Reset memory if it gets too large
    if interaction_count >= MAX_MEMORY_SIZE:
        memory.clear()
        interaction_count = 0
    
    response = agent_executor.invoke({"input": msg.text})
    
    interaction_count += 1  # Increase interaction count
    
    return {
        "response": response.get("output", "No response generated"),
        "intermediate_steps": response.get("intermediate_steps", [])
    }


