from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import prompts as p
import os
import agent_tools as tools
import schema 
from langgraph.graph import StateGraph, START, END

load_dotenv()
CONTEXT_SIZE= int(os.getenv('CONTEXT_SIZE'))
LOG_PATH= os.getenv('LOG_PATH')
DOC_PATH= os.getenv('LOG_PATH_MOD')

def process(state: schema.AgentState) -> schema.AgentState:
    """The agent processes the messages and updates the state"""    
    system_prompt= SystemMessage(content=p.SYSTEM_ASSISTANT)
    response= schema.llm.invoke([system_prompt]+state['messages'][-CONTEXT_SIZE:])
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
       print(f"TOOL CALLS: {[tc['name'] for tc in response.tool_calls]}")
       
    print(f"\nDORCAS: {response.content}")
    return {'messages': [response]}

def should_continue(state: schema.AgentState) -> str: # Returns a string path
    """Check if the last message contains any tool calls"""
    last_message = state['messages'][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return 'continue'
    return 'end'
   
def take_action(state: schema.AgentState) -> schema.AgentState:
    """The agent takes action based on the tool calls"""
    tool_calls= state['messages'][-1].tool_calls
    results= []
    for tool_call in tool_calls:
        print(f"Processing tool call: {tool_call['name']} with args: {tool_call['args']}")
        
        if not tool_call['name'] in tools.tools_dict:
            print(f"Tool {tool_call['name']} not found.")
            result= "Tool not found."
        else:
            result= tools.tools_dict[tool_call['name']].invoke(tool_call['args'])
            print(f"Tool result: {result}, {len(tool_calls)} steps.")
            
            results.append(ToolMessage(tool_call_id= tool_call['id'], 
                                       name= tool_call['name'], 
                                       content= result))
    print("Completed tool calls.")
    return {'messages': results}

def running_agent():
    print("\n INITIALIZING DORCAS")
    conversation_history= []
    if LOG_PATH is None:
        raise ValueError("LOG_PATH is not defined in .env")
    
    with open(LOG_PATH,'r') as file:
        for line in file:
            if line.startswith("AI"):
                conversation_history.append(AIMessage(content= line[3:]))
            if line.startswith("Human"):
                conversation_history.append(HumanMessage(content= line[6:]))    
    while True:
        user_input= input("Enter: ")
        if user_input.lower() == 'exit':
            break
        
        #convo flow
        message= HumanMessage(content= user_input)
        conversation_history.append(message)
        result= agent.invoke({'messages': conversation_history})
        response= result['messages'][-1]
        conversation_history.append(response)
        
        with open(LOG_PATH, "w") as file:
            file.write("conversation log\n")
            for message in conversation_history:
                if isinstance(message, HumanMessage):
                    file.write(f"Human: {message.content}\n")
                elif isinstance(message, AIMessage):
                    file.write(f"AI: {message.content}\n")
            file.write("log end\n")
            
    print("Conversation history saved to log.txt")
    
#graph
graph= StateGraph(schema.AgentState)
graph.add_node("llm", process)
graph.add_node("retriever", take_action)

graph.add_edge(START, "llm")
graph.add_conditional_edges('llm', should_continue,{
    'continue': "retriever",
    'end': END  
})
graph.add_edge("retriever", "llm")

agent= graph.compile()