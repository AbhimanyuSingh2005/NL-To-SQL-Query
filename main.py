import json
import requests
import time

total_tokens = 0

def load_input_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def generate_sqls(data):
    API_KEY = "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"
    MODEL = "llama3-8b-8192"

    sql_statements = []
    for entry in data:
        nl_query = entry.get("NL", "").strip()
        if not nl_query:
            sql_statements.append({"NL": nl_query, "Query": ""})
            continue

        prompt = f"Generate an SQL statement for the following NL query: {nl_query}"
        response = call_groq_api(API_KEY, MODEL, [{"role": "user", "content": prompt}])
        print("Response:", response)
        sql_response = response['choices'][0].get('message', {}).get('content', "Generated SQL") if response and 'choices' in response else "Error: Invalid API response"
        sql_statements.append({"NL": nl_query, "Query": sql_response})
    
    return sql_statements

def correct_sqls(sql_statements):
    API_KEY = "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"
    MODEL = "llama3-8b-8192"

    corrected_sqls = []
    for entry in sql_statements:
        incorrect_query = entry.get("IncorrectQuery", "").strip()
        if not incorrect_query:
            corrected_sqls.append({"IncorrectQuery": incorrect_query, "CorrectQuery": ""})
            continue

        prompt = f"Correct the following SQL query:\n{incorrect_query}"
        response = call_groq_api(API_KEY, MODEL, [{"role": "user", "content": prompt}])
        print("API Response:", response)
        corrected_query = response['choices'][0].get('message', {}).get('content', "Corrected SQL") if response and 'choices' in response else "Error: Invalid API response"
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
        'temperature': temperature,
        'max_tokens': max_tokens,
        'n': n
    }

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    total_tokens += response_json.get('usage', {}).get('completion_tokens', 0)
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
