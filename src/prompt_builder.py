# src/prompt_builder.py
def construct_prompt(new_query, retrieved_examples):
    """
    Construct a few-shot prompt using retrieved NL-to-SQL pairs.
    
    :param new_query: The new NL query.
    :param retrieved_examples: List of records, each with keys "NL" and "Query".
    :return: Prompt string.
    """
    prompt = "Below are some examples of NL-to-SQL pairs:\n\n"
    for ex in retrieved_examples:
        prompt += f"NL: {ex['NL']}\nSQL: {ex['Query']}\n\n"
    prompt += "Now, generate the SQL for the following NL query:\n"
    prompt += f"NL: {new_query}\nSQL:"
    return prompt
