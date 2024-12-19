from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from loader import load_documents, split_text, embedding
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
import asyncio, time
from tools import call_calculator, write_to_file, read_file, fetch_weather_data, solve_equation, add_two_integers, multiply_two_integers, fetch_current_local_time, search_for_information_in_documents_or_database
#nemotron-mini
#granite3-moe:3b
#granite3-dense
#qwen2.5:3b
#qwen2.5:3b-instruct-q2_K
with open('instructions.txt', 'r') as file:
  prompt_instructions = file.read()

# Initialize the Ollama client with the llama3 model
model = ChatOllama(model="qwen2.5:3b")

messages = [
    SystemMessage(content=prompt_instructions),
]


tools = [add_two_integers, multiply_two_integers, fetch_current_local_time,  call_calculator, write_to_file, read_file, fetch_weather_data, solve_equation, search_for_information_in_documents_or_database]
tools_dict = {
    "add_two_integers": add_two_integers,
    "multiply_two_integers": multiply_two_integers,
    "fetch_current_local_time": fetch_current_local_time,
    "call_calculator": call_calculator,
    "write_to_file": write_to_file,
    "read_file": read_file,
    "fetch_weather_data": fetch_weather_data,
    "solve_equation": solve_equation,
    "search_for_information_in_documents_or_database": search_for_information_in_documents_or_database
}


model_with_tools = model.bind_tools(tools)

def print_messages():
    for i in messages:
        print(i)
        print("=========")

def invoke_model():

    response = model_with_tools.invoke(messages)

    if response.content:
        print(response.content)
        messages.append(AIMessage(content=response.content))
    if response.tool_calls:
        print(f"Tools: {response.tool_calls}")
        print()
    return response

def invoke_tool_call(tool_calls):
    for tool in tool_calls:
        func_name = tool["name"].lower()
        args = tool["args"]

        selected_function = tools_dict.get(func_name)
        if selected_function:
            tool_msg = selected_function.invoke(tool)
        else:
            print(f"Function {func_name} not found")
        messages.append(tool_msg)


async def main():
    print("Loading model...")
    model_with_tools.invoke(messages)


    while True:
        print()
        input_text = input("User:  ")

        if input_text == "!print":
            print_messages()
            continue
        elif input_text == "exit":
            return
        messages.append(HumanMessage(content=input_text))

        print("Assistant: ", end="")        
        start = time.time()
        response = invoke_model()
        
        if response.tool_calls:
            invoke_tool_call(response.tool_calls)
            invoke_model()
        end = time.time()
        print(f"\n\n\033[91mTime taken was {end-start} seconds\033[0m")


if __name__ == "__main__":
    asyncio.run(main())


""""
        
        """""