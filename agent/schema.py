from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
import agent_tools as tools
from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools= [tools.retriever]
tools_dict= {tool.name: tool for tool in tools}
llm= ChatOllama(model= 'llama3.2:3B', temperature= 0.3)
rag_agent= llm
