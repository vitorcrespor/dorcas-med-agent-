from langgraph.graph import StateGraph, START, END
import state as s 
from langgraph.prebuilt import ToolNode

graph= StateGraph(s.AgentState)
graph.add_node("process", s.process)
tool_node= ToolNode(tools=s.tools)
graph.add_node("tool", tool_node)

graph.add_edge(START, "process")
graph.add_conditional_edges('process', s.should_continue,{
    'continue': "tool",
    'end': END  
})
graph.add_edge("tool", "process")

agent= graph.compile()