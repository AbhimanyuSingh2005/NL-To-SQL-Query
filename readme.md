🧠** AI-Powered SQL Query Generator and Error Corrector**
Team Codyaan – Abhimanyu Singh, Aryan Srivastava, Akshat Yadav

🚀 Overview
In enterprise environments, querying large and complex databases requires deep SQL expertise and domain knowledge. This project aims to bridge that gap by enabling users—technical or not—to interact with databases using natural language. Our AI-powered agent can:
- ✅ Convert natural language (NL) queries into accurate SQL statements
- 🔧 Automatically correct SQL queries with syntactic or semantic errors
- 🔍 Retrieve semantically similar examples to improve query generation

💡 Inspiration
- Frequent SQL errors during development cycles
- Difficulty faced by non-technical users in accessing and querying data
- The need for a modular, scalable, and easy-to-integrate solution
- Advances in LLMs (like GROQ) and vector search (FAISS) inspired our architecture

🧩 Key Features
- 🗣️ Natural Language to SQL: Converts user intent into executable SQL queries
- 🛠️ SQL Correction Module: Fixes queries with syntax issues or incorrect schema usage
- 🧠 Semantic Retrieval: Uses FAISS and SentenceTransformers to fetch similar examples
- 🔗 GROQ API Integration: Leverages powerful LLMs for precise query generation

⚙️ Tech Stack
| Component | Technology Used | 
| Language Model | GROQ API | 
| Embedding Model | sentence-transformers | 
| Vector Search Engine | faiss-cpu | 
| Backend Logic | Python (json, requests, os, numpy, time) | 



🔄 Workflow
graph TD
    A[User Input (Natural Language or SQL)] --> B[Preprocessing & Vectorization]
    B --> C[FAISS Vector Search]
    C --> D[Prompt Construction with Similar Examples]
    D --> E[GROQ API Call]
    E --> F[SQL Generation or Correction]
    F --> G[Output Saved to JSON]


Step-by-Step Breakdown
- Input Processing
- Load training and inference data from JSON files
- Vectorization
- Convert text to embeddings using SentenceTransformer
- Vector Database
- Store and retrieve embeddings using FAISS
- Load from disk if already built
- Prompt Construction
- Retrieve similar examples and build few-shot prompts
- GROQ API Call
- Send prompt to GROQ for SQL generation or correction
- Output Handling
- Save generated/corrected SQL to JSON for downstream use

📦 Installation
git clone https://github.com/Aryansrivastava07/NL_to_SQL.git
cd NL_to_SQL
pip install -r requirements.txt



🧪 Example Usage
from sql_agent import generate_sql

query = "Show me the top 5 customers by revenue in 2023"
sql_output = generate_sql(query)
print(sql_output)



📈 Impact
- ⏱️ Saves time and reduces human error in SQL writing
- 🧩 Modular and scalable architecture
- 🌍 Makes data access more inclusive for non-technical users

📣 Call to Action
We’re excited to see how this technology can evolve! Contributions, feedback, and collaborations are welcome to help build smarter, more intuitive data systems.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

