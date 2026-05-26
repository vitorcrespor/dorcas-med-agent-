import graph as g
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
LOG_PATH= os.getenv('LOG_PATH')

def main():
    if LOG_PATH is None:
        raise ValueError("LOG_PATH is not defined in .env")
    
    conversation_history= []
    with open(LOG_PATH,'r') as file:
        for line in file:
            if line.startswith("AI"):
                conversation_history.append(AIMessage(content= line[3:]))
            if line.startswith("Human"):
                conversation_history.append(HumanMessage(content= line[6:]))
    
    user_input= input("USER: ")
    
    while user_input!= 'exit':
        conversation_history.append(HumanMessage(content= user_input))
        result= g.agent.invoke({'messages': conversation_history})
        conversation_history= result['messages']
        user_input= input("USER: ")
    
    with open(LOG_PATH, "w") as file:
        file.write("conversation log\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"Human: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n")
        file.write("log end\n")
    print("Conversation history saved to log.txt")

if __name__ == "__main__":
     main()
