from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
import agent_tools as to
from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rag_query: str
    rag_context: str

tools= [to.retriever_tool]
tools_dict= {tools.name: tool for tool in tools}
lm= ChatOllama(model= 'llama3.2:3B', temperature= 0.3)
rag_agent= lm.bind(tools)