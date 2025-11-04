# import os
# os.environ["HIPPO_MODE"] = "online"
# from typing import List
# import json
# import argparse
# import logging
# import sys
# import multiprocessing
# # from src.hipporag import HippoRAG  <-- REMOVE THIS LINE
# sys.path.append(os.path.abspath("src"))

# def tests():
#     # Import HippoRAG *inside* the function
#     from src.hipporag import HippoRAG 
#     import logging

# # --- Enable file logging ---
#     logging.basicConfig(
#     level=logging.DEBUG,  # or INFO if you want fewer messages
#     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
#     handlers=[
#         logging.FileHandler("hipporag.log"),  # log file name
#         logging.StreamHandler()  # still show in console too
#     ]
# )
#     # Prepare datasets and evaluation
#     docs = [
#         "Oliver Badman is a politician.",
#         "George Rankin is a politician.",
#         "Thomas Marwick is a politician.",
#         "Cinderella attended the royal ball.",
#         "The prince used the lost glass slipper to search the kingdom.",
#         "When the slipper fit perfectly, Cinderella was reunited with the prince.",
#         "Erik Hort's birthplace is Montebello.",
#         "Marina is bom in Minsk.",
#         "Montebello is a part of Rockland County."
#     ]

#     save_dir = 'outputs/openai_test'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
#     llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
#     embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)

#     # Startup a HippoRAG instance
#     hipporag = HippoRAG(
#                         # global_config=None,
                        
#                         save_dir=save_dir,
#                         llm_base_url="https://api.openai.com/v1",
#                         embedding_base_url="https://api.openai.com/v1",
#                         llm_model_name=llm_model_name,
#                         embedding_model_name=embedding_model_name)

#     # Run indexing
#     hipporag.index(docs=docs)

#     # Separate Retrieval & QA
#     queries = [
#         "What is George Rankin's occupation?",
#         "How did Cinderella reach her happy ending?",
#         "What county is Erik Hort's birthplace a part of?"
#     ]

#     # For Evaluation
#     answers = [
#         ["Politician"],
#         ["By going to the ball."],
#         ["Rockland County"]
#     ]

#     gold_docs = [
#         ["George Rankin is a politician."],
#         ["Cinderella attended the royal ball.",
#          "The prince used the lost glass slipper to search the kingdom.",
#          "When the slipper fit perfectly, Cinderella was reunited with the prince."],
#         ["Erik Hort's birthplace is Montebello.",
#          "Montebello is a part of Rockland County."]
#     ]

#     print(hipporag.rag_qa(queries=queries,
#                          gold_docs=gold_docs,
#                          gold_answers=answers)[-2:])

    
# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     multiprocessing.set_start_method("spawn", force=True)
#     tests()

import os
import sys
import logging
import multiprocessing

# --- 1) Force online mode BEFORE importing hipporag ---
os.environ["HIPPO_MODE"] = "online"

# --- 2) Ensure src dir is on sys.path so Python can import your package ---
sys.path.append(os.path.abspath("src"))

# --- 3) Create logs folder and configure root logger BEFORE any imports that create loggers ---
os.makedirs("logs", exist_ok=True)

# Option A: Single file + console
logging.basicConfig(
    level=logging.DEBUG,  # change to INFO to reduce verbosity
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/hipporag.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Optional: enable debug only for hipporag package (less noisy)
# logging.getLogger("src.hipporag").setLevel(logging.DEBUG)

# --- 4) Now import HippoRAG (after logging configured) ---
from src.hipporag import HippoRAG

def tests():
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball."
    ]

    hippo = HippoRAG(
        save_dir="outputs/openai_test",
        llm_base_url="https://api.openai.com/v1",
        embedding_base_url="https://api.openai.com/v1",
        llm_model_name="gpt-4o-mini",
        embedding_model_name="text-embedding-3-small"
    )

    hippo.index(docs=docs)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    tests()
