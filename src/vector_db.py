import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.data_loader import load_json_file

def build_vector_database(dataset, prefix, field_name, model_name="all-MiniLM-L6-v2"):
    """
    Build (or load) a FAISS vector database for the given dataset.
    The function uses the specified field as the text basis for vectorization.
    Artifacts (index, embeddings, dataset) are saved in the project root using the prefix.
    
    :param dataset: List of records.
    :param prefix: Prefix string for filenames (e.g., "gen" or "corr").
    :param field_name: The key in each record to vectorize (e.g., "NL" or "IncorrectQuery").
    :param model_name: Name of the SentenceTransformer model.
    :return: Tuple (FAISS index, SentenceTransformer model, dataset)
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, "..")
    index_file = os.path.join(root_dir, f"{prefix}_vector_index.faiss")
    embeddings_file = os.path.join(root_dir, f"{prefix}_embeddings.npy")
    training_data_file = os.path.join(root_dir, f"{prefix}_training_data.json")
    
    model = SentenceTransformer(model_name)
    
    if os.path.exists(index_file) and os.path.exists(embeddings_file) and os.path.exists(training_data_file):
        print(f"Loading existing {prefix} vector database from disk...")
        index = faiss.read_index(index_file)
        embeddings = np.load(embeddings_file)
        with open(training_data_file, "r", encoding="utf-8") as f:
            training_data = json.load(f)
        return index, model, training_data
    else:
        print(f"Building {prefix} vector database...")
        texts = [record[field_name] for record in dataset if field_name in record and record[field_name].strip()]
        if not texts:
            raise ValueError(f"No valid '{field_name}' entries found in the dataset for prefix '{prefix}'.")
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        
        # Save the artifacts
        faiss.write_index(index, index_file)
        np.save(embeddings_file, embeddings)
        with open(training_data_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4)
        return index, model, dataset

if __name__ == "__main__":
    # For testing: build generation DB using "NL" from train_generate_task.json
    gen_data = load_json_file("train_generate_task.json")
    index, model, _ = build_vector_database(gen_data, prefix="gen", field_name="NL")
    print("Generation vector DB built. Total vectors:", index.ntotal)

    # For testing: build correction DB using "IncorrectQuery" from train_query_correction_task.json
    corr_data = load_json_file("train_query_correction_task.json")
    index, model, _ = build_vector_database(corr_data, prefix="corr", field_name="IncorrectQuery")
    print("Correction vector DB built. Total vectors:", index.ntotal)
