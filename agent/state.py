from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from matplotlib.style import context
import prompts as p
import os

load_dotenv()
CONTEXT_SIZE= int(os.getenv('CONTEXT_SIZE'))
LOG_PATH= os.getenv('LOG_PATH')
DOC_PATH= os.getenv('LOG_PATH_MOD')

doc_content= ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool 
def add(a: int, b: int) -> int:
    """Add two integers. Use only when the user explicitly asks for arithmetic addition."""
    return a + b

tools= [add]
llm= ChatOllama(model= 'llama3.2:3B')

def process(state: AgentState) -> AgentState:
    """The agent processes the messages and updates the state"""    
    system_prompt= SystemMessage(content=p.SYSTEM_ASSISTANT)
    response= llm.invoke([system_prompt]+state['messages'][-CONTEXT_SIZE:])
    if hasattr(response, 'tool_calls') and response.tool_calls:
       print(f"TOOL CALLS: {[tc['name'] for tc in response.tool_calls]}")
    print(f"\nAI: {response.content}")
    return {'messages': [response]}

def should_continue(state: AgentState) -> AgentState:
    """The agent decides whether to continue processing or not"""
    messages= state['messages']
    last_message= messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else: return 'continue'