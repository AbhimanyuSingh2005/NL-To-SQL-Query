import json
import requests
import time

total_tokens = 0

from src.data_loader import load_json_file, load_input_file
from src.vector_db import build_vector_database
from src.prompt_builder import build_prompt_with_retrieval

def generate_sqls(data):
    API_KEY = "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"
    MODEL = "llama3-8b-8192"

    gen_training_data = load_json_file("train_generate_task.json")
    index, embed_model, gen_training_data = build_vector_database(gen_training_data, prefix="gen", field_name="NL", model_name="all-MiniLM-L6-v2")
    
    sql_statements = []
    for entry in data:
        nl_query = entry.get("NL", "").strip()
        if not nl_query:
            sql_statements.append({"NL": nl_query, "Query": ""})
            continue
        
        prompt = build_prompt_with_retrieval(nl_query, index, embed_model, gen_training_data, k=3)
        response = call_groq_api(API_KEY, MODEL, [{"role": "user", "content": prompt}])
        print("Response:", response)
        if response and 'choices' in response:
            sql_response = response['choices'][0].get('message', {}).get('content', "Generated SQL")
        else:
            sql_response = "Error: Invalid API response"
        sql_statements.append({"NL": nl_query, "Query": sql_response})
    
    return sql_statements

def correct_sqls(data):
    API_KEY = "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"
    MODEL = "llama3-8b-8192"

    corr_training_data = load_json_file("train_query_correction_task.json")
    index, embed_model, corr_training_data = build_vector_database(corr_training_data, prefix="corr", field_name="IncorrectQuery", model_name="all-MiniLM-L6-v2")
    
    corrected_sqls = []
    for entry in data:
        incorrect_query = entry.get("IncorrectQuery", "").strip()
        nl_query = entry.get("NL", "").strip() or incorrect_query
        if not incorrect_query:
            corrected_sqls.append({"IncorrectQuery": incorrect_query, "CorrectQuery": ""})
            continue

        query_embedding = embed_model.encode([nl_query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, 3)
        retrieved_examples = [corr_training_data[i] for i in indices[0]]
        
        prompt = "Below are examples of NL queries with their incorrect and corrected SQL queries:\n\n"
        for ex in retrieved_examples:
            field = ex["NL"] if "NL" in ex else ex["IncorrectQuery"]
            prompt += f"NL: {field}\nIncorrect SQL: {ex['IncorrectQuery']}\nCorrect SQL: {ex['CorrectQuery']}\n\n"
        prompt += f"Now, correct the following SQL query:\nNL: {nl_query}\nIncorrect SQL: {incorrect_query}\nCorrect SQL:"
        
        response = call_groq_api(API_KEY, MODEL, [{"role": "user", "content": prompt}])
        print("API Response:", response)
        if response and 'choices' in response:
            corrected_query = response['choices'][0].get('message', {}).get('content', "Corrected SQL")
        else:
            corrected_query = "Error: Invalid API response"
        corrected_sqls.append({"IncorrectQuery": incorrect_query, "CorrectQuery": corrected_query})
    
    return corrected_sqls

def call_groq_api(api_key, model, messages, temperature=0.0, max_tokens=1000, n=1):
    global total_tokens
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    total_tokens += response_json.get('usage', {}).get('completion_tokens', 0)
    print(f"\nTotal tokens: {total_tokens}\n\n")
    return response_json

def main():
    input_file_path_1 = 'data/input_file_for_sql_generation.json'
    input_file_path_2 = 'data/input_file_for_sql_correction.json'
    
    data_1 = load_input_file(input_file_path_1)
    data_2 = load_input_file(input_file_path_2)
    
    start = time.time()
    sql_statements = generate_sqls(data_1)
    generate_sqls_time = time.time() - start
    
    start = time.time()
    corrected_sqls = correct_sqls(data_2)
    correct_sqls_time = time.time() - start
    
    assert len(data_2) == len(corrected_sqls)
    assert len(data_1) == len(sql_statements)
    
    with open('output_sql_correction_task.json', 'w') as f:
        json.dump(corrected_sqls, f)
    
    with open('output_sql_generation_task.json', 'w') as f:
        json.dump(sql_statements, f)
    
    return generate_sqls_time, correct_sqls_time

if __name__ == "__main__":
    generate_sqls_time, correct_sqls_time = main()
    print(f"Time taken to generate SQLs: {generate_sqls_time} seconds")
    print(f"Time taken to correct SQLs: {correct_sqls_time} seconds")
    print(f"Total tokens: {total_tokens}")
