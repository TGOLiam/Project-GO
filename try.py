import ollama

with open('instructions.txt', 'r') as file:
  prompt_instructions = file.read()

MODEL = 'llama3.1'

def fetch_current_local_time() -> str:
    """
    Retrieves the current local time as a formatted string.

    Designed for scenarios where the local time is needed,
    such as in interactive systems or when specifically requested by users.
    """


    import datetime
    import asyncio

    x = datetime.datetime.now()
    formatted_time = x.strftime("%Y-%m-%d %I:%M %p")
    date = x.strftime("%Y-%m-%d")
    weekday = x.strftime("%A")
    return f"Time is {formatted_time}; Date is {date}; Weekday is {weekday}"

def multiply_two_integers(a: int, b: int) -> int:
    """
    Multiplies two integers and returns the result.

    Args:
        a: The first integer to be multiplied.
        b: The second integer to be multiplied.

    Useful for basic calculations or when a product is specifically needed.
    """
    return a * b

messages = [
   {'role': 'system', 'content': prompt_instructions}
]

while True:
    prompt = input("Enter: ")

    messages.append({'role': 'user', 'content': prompt})


    response = ollama.chat(
        model= MODEL,
        messages=messages,
        tools=[fetch_current_local_time]
    )

    messages.append({'role': 'assistant', 'content': response.message.content})
    if response.message.tool_calls: 
        
        messages.append({'role': 'assistant', 'content': "TIme is 7am"})


    print(response)
    print()
    print(response.message.content)
    print(response.message.tool_calls)