from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import agent_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from operator import add
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    summary: str
    
tools_list= [agent_tools.retriever_tool, agent_tools.pubmed_tool, agent_tools.fhirpath_tool]
tools_dict= {tool.name: tool for tool in tools_list}
lm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
                            temperature=0.3,)
rag_agent= lm.bind_tools(tools_list)

