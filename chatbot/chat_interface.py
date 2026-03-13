from chains.rag_chain import get_rag_chain


def run_cli():
    chain = get_rag_chain()
    print("RAG Chatbot. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        result = chain.invoke({"input": query})
        answer = result.get("answer", "")
        print(f"Assistant: {answer}\n")
