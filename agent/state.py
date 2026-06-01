from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import prompts as p
import os
import agent_tools as to
import schema 
from langgraph.graph import StateGraph, START, END
import support.functions as f
from langgraph.errors import GraphRecursionError

load_dotenv()
CONTEXT_SIZE= int(os.getenv('CONTEXT_SIZE'))
LOG_PATH= os.getenv('LOG_PATH')
DOC_PATH= os.getenv('LOG_PATH_MOD')
TOOL_LOG_PATH= os.getenv('TOOL_LOG_PATH')

async def process(state: schema.AgentState) -> schema.AgentState:
    """The agent processes the messages using recent messages."""
    recent_messages= state["messages"][-CONTEXT_SIZE:]
    system_prompt= SystemMessage(content= p.SYSTEM_PROMPT)
    response= await schema.rag_agent.ainvoke([system_prompt] + recent_messages)

    print(f"\nDORCAS: {f.content_to_text(response.content)}")
    return {"messages": [response]}

def should_continue(state: schema.AgentState) -> str: # Returns a string path
    """Check if the last message contains any tool calls"""
    last_message= state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'continue'
    return 'end'
   
async def take_action(state: schema.AgentState) -> schema.AgentState:
    """The agent takes action based on the tool calls"""
    tool_calls= state['messages'][-1].tool_calls
    results= []
    for tool_call in tool_calls:
        print(f"Processing tool call: {tool_call['name']} with args: {tool_call['args']}")
        
        if not tool_call['name'] in schema.tools_dict:
            print(f"Tool {tool_call['name']} not found.")
            result= "Tool not found."
        else:
            try:
                result = await schema.tools_dict[tool_call['name']].ainvoke(tool_call['args'])
            except Exception as e:
                result = f"Tool error: {type(e).__name__}: {e}"     
                       
            print(f"Tool result: {result}, {len(tool_calls)} steps.")
            results.append(ToolMessage(tool_call_id= tool_call['id'], 
                                       name= tool_call['name'], 
                                       content= result))
            
    print("Completed tool calls.")
    return {'messages': results}

async def running_agent():
    print("\n INITIALIZING DORCAS")
    if LOG_PATH is None:
        raise ValueError("LOG_PATH is not defined in .env")
    
    conversation_history= f.log_ingestion()
    
    while True:
        user_input= input("USER: ")
        if user_input.lower() == 'exit':
            break
        
        #convo flow
        message= HumanMessage(content= user_input)
        conversation_history.append(message)
        result= await agent.ainvoke({
            "messages": conversation_history})

        conversation_history= result["messages"]
          
        f.log(conversation_history)
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