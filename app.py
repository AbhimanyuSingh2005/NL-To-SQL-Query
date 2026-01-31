import streamlit as st
import os
import json

# Set page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="NL to SQL Assistant",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ NL to SQL Query Assistant")
st.markdown("Generate or Correct SQL queries using Natural Language.")

# Late imports to allow UI to render first
with st.spinner("Initializing application and loading modules..."):
    import main
    from src.data_loader import load_json_file
    from src.vector_db import build_vector_database
    from src.prompt_builder import build_prompt_with_retrieval

# --- Caching Resources ---
@st.cache_resource
def load_gen_resources():
    """Load resources for SQL Generation."""
    # Ensure we can find the data file before trying to build/load
    # Note: data_loader.load_json_file looks in ../data relative to itself.
    # We trust it finds 'train_generate_task.json'.
    
    gen_data = load_json_file("train_generate_task.json")
    index, embed_model, gen_data = build_vector_database(
        gen_data, 
        prefix="gen", 
        field_name="NL", 
        model_name="all-MiniLM-L6-v2"
    )
    return index, embed_model, gen_data

@st.cache_resource
def load_corr_resources():
    """Load resources for SQL Correction."""
    corr_data = load_json_file("train_query_correction_task.json")
    index, embed_model, corr_data = build_vector_database(
        corr_data, 
        prefix="corr", 
        field_name="IncorrectQuery", 
        model_name="all-MiniLM-L6-v2"
    )
    return index, embed_model, corr_data

# --- API Configuration ---
st.sidebar.subheader("Configuration")
default_key = os.environ.get("GROQ_API_KEY", "")
api_key_input = st.sidebar.text_input("Groq API Key", value=default_key, type="password", help="Get your key from https://console.groq.com/keys")

if not api_key_input:
    st.info("Please enter your Groq API Key in the sidebar to proceed.")
    st.stop()

API_KEY = api_key_input
MODEL = "llama-3.1-8b-instant"

# Sidebar for navigation
task = st.sidebar.radio("Select Task", ["Generate SQL", "Correct SQL"])

if task == "Generate SQL":
    st.header("Generate SQL from Natural Language")
    st.markdown("Enter a natural language description of the data you want to retrieve.")
    
    # Load resources once
    with st.status("Loading models and vector DB...", expanded=True) as status:
        try:
            st.write("Loading generation resources...")
            index, embed_model, gen_training_data = load_gen_resources()
            status.update(label="Resources loaded successfully!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Failed to load resources", state="error")
            st.error(f"Failed to load resources: {e}")
            st.stop()

    nl_query = st.text_area("Natural Language Query", height=150, placeholder="e.g., Show me all users who signed up last month.")
    
    if st.button("Generate SQL"):
        if nl_query.strip():
            with st.spinner("Generating SQL..."):
                try:
                    # Logic adapted from main.generate_sqls
                    prompt = build_prompt_with_retrieval(nl_query, index, embed_model, gen_training_data, k=3)
                    
                    response = main.call_groq_api(API_KEY, MODEL, [{"role": "user", "content": prompt}])
                    
                    if response and 'choices' in response:
                        sql_response = response['choices'][0].get('message', {}).get('content', "Generated SQL")
                        st.subheader("Generated SQL:")
                        st.code(sql_response, language="sql")
                        
                        # Optional: Show debug info
                        with st.expander("View Prompt & Details"):
                            st.text(prompt)
                            st.json(response['usage'])
                    else:
                        st.error("Error: Invalid API response")
                        st.json(response)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

elif task == "Correct SQL":
    st.header("Correct Invalid SQL Queries")
    st.markdown("Provide an incorrect SQL query and (optionally) the original natural language intent to get a fixed version.")
    
    # Load resources once
    with st.status("Loading models and vector DB...", expanded=True) as status:
        try:
            st.write("Loading correction resources...")
            index, embed_model, corr_training_data = load_corr_resources()
            status.update(label="Resources loaded successfully!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Failed to load resources", state="error")
            st.error(f"Failed to load resources: {e}")
            st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        nl_intent = st.text_area("Natural Language Intent (Optional)", height=150, placeholder="What was this query supposed to do?")
    with col2:
        incorrect_sql = st.text_area("Incorrect SQL Query", height=150, placeholder="SELECT * FROM table WHERE ...")
        
    if st.button("Correct SQL"):
        if incorrect_sql.strip():
            with st.spinner("Correcting SQL..."):
                try:
                    # Logic adapted from main.correct_sqls
                    nl_query = nl_intent.strip() or incorrect_sql.strip()
                    
                    query_embedding = embed_model.encode([nl_query], convert_to_numpy=True)
                    distances, indices = index.search(query_embedding, 3)
                    retrieved_examples = [corr_training_data[i] for i in indices[0]]
                    
                    prompt = "Below are examples of NL queries with their incorrect and corrected SQL queries:\n\n"
                    for ex in retrieved_examples:
                        # Handle potential key variations if necessary, matching main.py logic
                        field = ex["NL"] if "NL" in ex else ex["IncorrectQuery"]
                        prompt += f"NL: {field}\nIncorrect SQL: {ex['IncorrectQuery']}\nCorrect SQL: {ex['CorrectQuery']}\n\n"
                    
                    prompt += f"Now, correct the following SQL query:\nNL: {nl_query}\nIncorrect SQL: {incorrect_sql}\nCorrect SQL:"
                    
                    response = main.call_groq_api(API_KEY, MODEL, [{"role": "user", "content": prompt}])
                    
                    if response and 'choices' in response:
                        corrected_query = response['choices'][0].get('message', {}).get('content', "Corrected SQL")
                        st.subheader("Corrected SQL:")
                        st.code(corrected_query, language="sql")
                        
                        with st.expander("View Prompt & Details"):
                            st.text(prompt)
                            st.json(response['usage'])
                    else:
                        st.error("Error: Invalid API response")
                        st.json(response)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter the incorrect SQL query.")

# Footer
st.markdown("---")
st.markdown("Powered by Groq, FAISS, and SentenceTransformers.")
