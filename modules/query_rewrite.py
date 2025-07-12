import os
import re

def rewrite_query(query: str, available_docs: list) -> str:
    doc_title = os.path.splitext(available_docs[0])[0].replace("_", " ").title()
    clean_query = query.lower().strip()
    clean_query = re.sub(r'[^\w\s]', '', clean_query)

    if "what happened" in clean_query or "summary" in clean_query or "explain" in clean_query:
        return f"What happened in the story {doc_title}?"

    if "theme" in clean_query or "moral" in clean_query or "lesson" in clean_query:
        return f"What is the theme of the story {doc_title}?"

    if "who died" in clean_query:
        return f"Who died in the story {doc_title}?"

    if clean_query.startswith(("who", "what", "when", "why", "how")):
        return f"{query.strip().capitalize()} in the story {doc_title}?"

    return query.strip().capitalize()
