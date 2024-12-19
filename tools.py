from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="mxbai-embed-large",)
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma")



















@tool
def call_calculator(expression: str) -> int:
    """
    Evaluates a simple mathematical expression and returns the result.
    Args:
        expression: A string representing a mathematical expression (e.g., '12 + 8').

    Useful for basic arithmetic operations like addition, subtraction, multiplication, etc.
    """
    return eval(expression)


@tool
def write_to_file(content: str, file_name: str = "output.txt") -> str:
    """
    Writes content to a specified file.
    Args:
        content: The content to be written to the file.
        file_name: The name of the file to write to (default is 'output.txt').

    Useful for saving text-based data to a file.
    """
    with open(file_name, "w") as file:
        file.write(content)
    return f"Written '{content}' to {file_name}"

@tool
def read_file(file_name: str = "output.txt") -> str:
    """
    Reads content from a specified file.
    Args:
        file_name: The name of the file to read from (default is 'output.txt').

    Useful for retrieving text-based data from a file.
    """
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."

@tool
def fetch_weather_data(location: str) -> dict:
    """
    Fetches weather data for a specified location.
    Args:
        location: The name of the location for which to retrieve weather data.

    Useful for providing real-time weather information for specific locations.
    """
    # Simulate an API response (dummy data for illustration)
    return {"location": location, "temperature": "22°C", "condition": "Clear"}

@tool
def solve_equation(equation: str) -> str:
    """
    Solves a simple mathematical equation and returns the result.
    Args:
        equation: A string representing an equation (e.g., 'x + 5 = 12').

    Useful for solving basic algebraic equations.
    """
    if equation == "x + 5 = 12":
        return "x = 7"
    return "Equation not recognized"

@tool
def multiply_two_integers(a: int, b: int) -> int:
    """
    Multiplies two integers and returns the result.
    Args:
        a: The first integer to be multiplied.
        b: The second integer to be multiplied.

    Useful for basic calculations or when a product is specifically needed.
    """
    return a * b

@tool
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

@tool
def add_two_integers(a: int, b: int) -> int:
    """
    Adds two integers and returns the result.

    Args:
        a: The first integer to be added.
        b: The second integer to be added.

    Suitable for simple arithmetic operations or when a sum is requested.
    """
    return a + b

@tool
def search_for_information_in_documents_or_database(query: str, k: int = 3) -> dict:
    """
    Searches the vector store (documents or database) to find the top-k most relevant pieces of information based on the input query, 
    and returns the results along with their confidence scores.

    Args:
        vector_store: The vector store instance containing the stored vectors or data for search (could be documents or a database).
        query: The input query text for which relevant information will be retrieved.
        k: The number of top results to return (default is 3).

    Returns:
        A dictionary containing:
            - "results": A list of the top-k most relevant information (text or document) retrieved from the vector store.
            - "confidence": A list of the similarity scores corresponding to each of the retrieved results.
    
    This function is ideal for querying a vector store to retrieve relevant data, helping the LLM find specific information such as addresses, facts, or other details based on natural language queries.
    """
    # Perform similarity search with score to get the top-k relevant results
    results = vector_store.similarity_search_with_score(query, k)
    
    # Separate results and confidence scores
    texts = [result[0] for result in results]
    scores = [result[1] for result in results]

    # Return both results and their corresponding confidence scores
    return {"results": texts, "confidence": scores}
