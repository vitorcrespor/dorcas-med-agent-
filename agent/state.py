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
llm= ChatOllama(model= 'llama3.2:3B').bind_tools(tools)

def setup(state: AgentState) -> AgentState:
    """The agent sets up the initial state"""
    load_messages= []
    with open(LOG_PATH,'r') as file:
        for line in file:
            if line.startswith("AI"):
                load_messages.append(AIMessage(content= line[3:]))
            if line.startswith("Human"):
                load_messages.append(HumanMessage(content= line[6:]))
    return {'messages': load_messages}

def process(state: AgentState) -> AgentState:
    """The agent processes the messages and updates the state"""    
    system_prompt= SystemMessage(content=p.SYSTEM_ASSISTANT)
    response= llm.invoke([system_prompt]+state['messages'][-CONTEXT_SIZE:])
    print("CURRENT STATE",state['messages'])
    #if hasattr(response, 'tool_calls') and response.tool_calls:
    #   print(f"TOOL CALLS: {[tc['name'] for tc in response.tool_calls]}")
    print(f"\nAI: {response.content}")
    return {'messages': [response]}

def should_continue(state: AgentState) -> AgentState:
    """The agent decides whether to continue processing or not"""
    messages= state['messages']
    last_message= messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else: return 'continue'