def build_prompt_with_retrieval(new_query, index, embed_model, training_data, k=3):
    """
    Construct a few-shot prompt using vector retrieval.
    
    This function encodes the new query, retrieves the top-k similar examples from the training data
    using the provided FAISS index and SentenceTransformer model, and then constructs a prompt
    that includes those examples and the new query.
    
    :param new_query: The new NL query (string).
    :param index: A FAISS index built from training embeddings.
    :param embed_model: The SentenceTransformer model used to encode text.
    :param training_data: The original training data (list of dictionaries), where each record has keys "NL" and "Query".
    :param k: Number of similar examples to retrieve (default is 3).
    :return: A prompt string.
    """
    # Encode the new query to a vector:
    query_embedding = embed_model.encode([new_query], convert_to_numpy=True)
    # Retrieve top-k similar examples using FAISS:
    distances, indices = index.search(query_embedding, k)
    retrieved_examples = [training_data[i] for i in indices[0]]
    
    # Build the prompt with few-shot examples:
    prompt = "Below are some examples of NL-to-SQL pairs:\n\n"
    for ex in retrieved_examples:
        prompt += f"NL: {ex['NL']}\nSQL: {ex['Query']}\n\n"
    prompt += "Now, generate the SQL for the following NL query. Reply with ONLY the SQL query and nothing else.\n"
    prompt += f"NL: {new_query}\nSQL:"
    return prompt

if __name__ == "__main__":
    # Simple test of the function.
    from sentence_transformers import SentenceTransformer
    import faiss, json, os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_data_path = os.path.join(current_dir, "..", "data", "train_generate_task.json")
    with open(training_data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    # Extract texts for vectorization.
    texts = [record["NL"] for record in training_data if "NL" in record and record["NL"].strip()]
    
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    new_query = "List all active customers who signed up last month."
    prompt = build_prompt_with_retrieval(new_query, index, embed_model, training_data, k=3)
    print("Constructed Prompt:\n", prompt)
