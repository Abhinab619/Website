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

# Initialize Vector Store & Retriever
vectorstore = Chroma(
    persist_directory=r"C:\Users\HP\Desktop\Projects 2k24\NS Apps- Wooorkkk\YOJNA_Query_classification\USING RAG\data_store",
    embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key='AIzaSyBgdymDNQMdnSEad-xYapzh1hS3F6wmxfE')
)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.7})

retriever_tool = create_retriever_tool(
    retriever=retriever, name="Udhami_Yojna", description="Information on this Yojna"
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

# Define tools
tools = [retriever_tool, direct_llm_answer]

chat_prompt_template = hub.pull("hwchase17/openai-tools-agent")

# Create tool calling agent
agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=chat_prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

@app.post("/chat")
def chat_with_model(msg: Message):
    response = agent_executor.invoke({"input": msg.text})
    return {"response": response["output"]}
