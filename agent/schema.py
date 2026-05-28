from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
import agent_tools
from operator import add


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    
tools_list= [agent_tools.retriever_tool, agent_tools.pubmed]
tools_dict= {tool.name: tool for tool in tools_list}
lm= ChatOllama(model= 'llama3.2:3B', temperature= 0.3)
rag_agent= lm.bind_tools(tools_list)

