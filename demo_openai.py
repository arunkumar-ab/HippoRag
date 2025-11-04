import os
import json
import logging
from typing import List
import argparse
import sys

sys.path.append(os.path.abspath("src"))

# 💡 Move all code inside a function
def run_test():
    from src.hipporag import HippoRAG

    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'outputs/local_test'
    llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_base_url= "https://api.openai.com/v1",
                        embedding_base_url= "https://api.openai.com/v1",
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name)

    hipporag.index(docs=docs)

    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        [
            "Cinderella attended the royal ball.",
            "The prince used the lost glass slipper to search the kingdom.",
            "When the slipper fit perfectly, Cinderella was reunited with the prince."
        ],
        [
            "Erik Hort's birthplace is Montebello.",
            "Montebello is a part of Rockland County."
        ]
    ]

    print(hipporag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=answers)[-2:])

    print(f"[DEBUG] Working directory: {hipporag.working_dir}")
    print(f"[DEBUG] Graph will be saved at: {hipporag._graph_pickle_filename}")

# 🛡️ Put all multiprocessing or model-loading logic inside this block
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Important for Windows
    run_test()