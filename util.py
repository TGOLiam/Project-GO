

BLUE = "\033[34m"
RESET = "\033[0m"



while True:

    print()

    input_text = input(BLUE + "Enter: " + RESET)

    results = vector_store.similarity_search(input_text, k=3)

    for doc, score in results:
        print("=========================================")
        print(f"Score: {score}\n")
        print(doc.page_content)
        print("=========================================")

