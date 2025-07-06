# 🧠 AI-Powered SQL Query Generator & Error Corrector  
### Team Codyaan – Abhimanyu Singh, Aryan Srivastava, Akshat Yadav  

---

## 🚀 Overview

In enterprise environments, querying large and complex databases requires deep SQL expertise and domain knowledge. This project bridges that gap by enabling users—technical or not—to interact with databases using natural language.

Our AI-powered agent can:

- ✅ Convert natural language (NL) queries into accurate SQL statements  
- 🔧 Automatically correct SQL queries with syntactic or semantic errors  
- 🔍 Retrieve semantically similar examples to improve query generation  

---

## 💡 Inspiration

- Frequent SQL errors during development cycles  
- Difficulty faced by non-technical users in accessing and querying data  
- The need for a modular, scalable, and easy-to-integrate solution  
- Advances in LLMs (like GROQ) and vector search (FAISS) inspired our architecture  

---

## 🧩 Key Features

- 🗣️ **Natural Language to SQL**: Converts user intent into executable SQL queries  
- 🛠️ **SQL Correction Module**: Fixes queries with syntax issues or incorrect schema usage  
- 🧠 **Semantic Retrieval**: Uses FAISS and SentenceTransformers to fetch similar examples  
- 🔗 **GROQ API Integration**: Leverages powerful LLMs for precise query generation  

---

## ⚙️ Tech Stack

| Component              | Technology Used              |
|------------------------|------------------------------|
| Language Model         | GROQ API                     |
| Embedding Model        | `sentence-transformers`      |
| Vector Search Engine   | `faiss-cpu`                  |
| Backend Logic          | Python (`json`, `requests`, `os`, `numpy`, `time`) |

---

## 🔄 Workflow

![Workflow Diagram]([https://your-image-link-here.com/workflow.png](https://drive.google.com/file/d/1_Ijma7M6hOPWAiHtJnSzjDkrQFEUdJKa/view?usp=sharing))  
<sub><i>Workflow</i></sub>

### Step-by-Step Breakdown

1. **Input Processing**  
   - Load training and inference data from JSON files  

2. **Vectorization**  
   - Convert text to embeddings using SentenceTransformer  

3. **Vector Database**  
   - Store and retrieve embeddings using FAISS  
   - Load from disk if already built  

4. **Prompt Construction**  
   - Retrieve similar examples and build few-shot prompts  

5. **GROQ API Call**  
   - Send prompt to GROQ for SQL generation or correction  

6. **Output Handling**  
   - Save generated/corrected SQL to JSON for downstream use  

---

## 📦 Installation

```bash
git clone https://github.com/Aryansrivastava07/NL_to_SQL.git
cd NL_to_SQL
pip install -r requirements.txt



🧪 Example Usage
from sql_agent import generate_sql

query = "Show me the top 5 customers by revenue in 2023"
sql_output = generate_sql(query)
print(sql_output)



📈 Impact
- ⏱️ Saves time and reduces human error in SQL writing
- 🧩 Modular and scalable architecture
- 🌍 Makes data access more inclusive for non-technical users

🤝 Contributing
We welcome contributions! Feel free to fork the repo, submit issues, or open pull requests to improve the system.

📣 Call to Action
We’re excited to see how this technology can evolve! Contributions, feedback, and collaborations are welcome to help build smarter, more intuitive data systems.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.


