from langchain_ollama import OllamaLLM

def query_ollama(system_prompt, task_prompt, model_selection, temperature):

    print("Using Ollama:", model_selection)
    llm = OllamaLLM(
        model=model_selection,
        temperature=temperature
    )
    result = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt}
    ])
    return result

def ollama_obj(model_selection, temperature):

    print("Using Ollama for obj:", model_selection)
    llm = OllamaLLM(
        model=model_selection,
        temperature=temperature
    )
    return llm

# testing
print(query_ollama("Sarcastic Assistant","Why is the sky blue?", "llama3.2", 0.7))