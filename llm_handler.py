from langchain.llms import ctransformers

def load_llm():
    
    print("Loading LLM...")
    llm = ctransformers(
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_type="llama",
        config={'max_new_tokens': 512, 'temperature': 0.1}
    )
    return llm