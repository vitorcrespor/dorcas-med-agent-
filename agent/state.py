from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import prompts as p
import os
import agent_tools as to
import schema 
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import support.functions as f

load_dotenv()
CONTEXT_SIZE= int(os.getenv('CONTEXT_SIZE'))
LOG_PATH= os.getenv('LOG_PATH')
DOC_PATH= os.getenv('LOG_PATH_MOD')
TOOL_LOG_PATH= os.getenv('TOOL_LOG_PATH')

def process(state: schema.AgentState) -> schema.AgentState:
    """The agent processes the messages using recent messages + summary."""
    recent_messages = state["messages"][-CONTEXT_SIZE:]
    summary = state.get("summary", "")
    system_prompt = SystemMessage(content=f"""{p.SYSTEM_PROMPT}
                                    Conversation summary so far:
                                    {summary if summary else "No previous summary yet."}
                                    Use the summary as long-term context.
                                    Use the recent messages as the immediate conversation context.
                                    """)

    response = schema.rag_agent.invoke([system_prompt] + recent_messages)

    print(f"\nDORCAS: {response.content}")
    new_summary = f.update_summary(
        old_summary=summary,
        recent_messages=recent_messages,
        ai_response=response,)

    return {"messages": [response],
            "summary": new_summary,}



def should_continue(state: schema.AgentState) -> str: # Returns a string path
    """Check if the last message contains any tool calls"""
    last_message= state['messages'][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'continue'
    return 'end'
   
def take_action(state: schema.AgentState) -> schema.AgentState:
    """The agent takes action based on the tool calls"""
    tool_calls= state['messages'][-1].tool_calls
    results= []
    for tool_call in tool_calls:
        print(f"Processing tool call: {tool_call['name']} with args: {tool_call['args']}")
        state['action_memory'].append(ToolMessage(tool_call))
        
        if not tool_call['name'] in to.tools_dict:
            print(f"Tool {tool_call['name']} not found.")
            result= "Tool not found."
        else:
            result= schema.tools_dict[tool_call['name']].invoke(tool_call['args'])
            print(f"Tool result: {result}, {len(tool_calls)} steps.")
            
            results.append(ToolMessage(tool_call_id= tool_call['id'], 
                                       name= tool_call['name'], 
                                       content= result))
            
    state['messages'].append(results)
    print("Completed tool calls.")
    return {'messages': results}

def running_agent():
    
    print("\n INITIALIZING DORCAS")
    conversation_history= []
    if LOG_PATH is None:
        raise ValueError("LOG_PATH is not defined in .env")
    
    with open(LOG_PATH,'r') as file:
        for line in file:
            if line.startswith("DORCAS"):
                conversation_history.append(AIMessage(content= line[3:]))
            if line.startswith("USER"):
                conversation_history.append(HumanMessage(content= line[6:]))    
    while True:
        user_input= input("Enter: ")
        if user_input.lower() == 'exit':
            break
        
        #convo flow
        message= HumanMessage(content= user_input)
        conversation_history.append(message)
        result = agent.invoke({
            "messages": conversation_history,
            "action_memory": tool_history})

        conversation_history = result["messages"]
        tool_history = result.get("action_memory", [])
        response = conversation_history[-1]
        print(f"\nDORCAS: {response.content}")  
          
        f.log(tool_history, conversation_history)
    print("Conversation history saved to log.txt")
    
#graph
graph= StateGraph(schema.AgentState)
graph.add_node("llm", process)
graph.add_node("tools", take_action)

graph.add_edge(START, "llm")
graph.add_conditional_edges('llm', should_continue,{
    'continue': "tools",
    'end': END  
})
graph.add_edge("tools", "llm")

agent= graph.compile()