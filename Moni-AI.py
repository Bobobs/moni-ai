"""
Moni‑AI
----------
A Retrieval‑Augmented‑Generation (RAG) assistant for rapid psychiatric
classification and reasoning.

Revision: **Ollama‑LLM Support** (May 2025)
==========================================
✓ Primary LLM backend is now **Ollama** for fully local inference
  • Default model: `llama3` (already pulled via `ollama pull llama3`)
  • Override with `--ollama-model MYMODEL` or `OLLAMA_MODEL` env var
✓ HuggingFace transformers pipeline kept as optional fallback (`--hf-model`)
✓ FAISS index persistence & safe deserialization unchanged
✓ Current LangChain callback API (StdOutCallbackHandler & LangChainTracer)
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse
import os
import sys

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer

# --- LLM backends -----------------------------------------------------------
from langchain_community.llms import Ollama  # local Ollama
from langchain.llms import HuggingFacePipeline  
import torch
import transformers



# ---------------------------------------------------------------------------
# ARGUMENTS & CONFIG
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Moni‑AI")
parser.add_argument("cli", nargs="?", help="Run in terminal CLI mode")
parser.add_argument("--ollama-model", dest="ollama_model", help="Ollama model name (default llama3)")
parser.add_argument("--ollama-url", dest="ollama_url", help="Ollama server URL (default http://localhost:11434)")
parser.add_argument("--hf-model", dest="hf_model", help="Optional HF model path/id for transformers fallback")
parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of FAISS index")
args = parser.parse_args()

DATA_FILE = Path("data/psy_reference.txt")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = args.ollama_model or os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = args.ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
HF_MODEL_ID = args.hf_model or os.getenv("HF_FALLBACK_MODEL")
TEMPERATURE = 0.25
MAX_NEW_TOKENS = 512
INDEX_PATH = Path("data/faiss_index")  # persisted vector store

# ---------------------------------------------------------------------------
# LOAD / REBUILD VECTOR STORE (PERSISTENT & SAFE)
# ---------------------------------------------------------------------------
print(f"[Moni-AI] Loading reference data from {DATA_FILE.resolve()}")
loader = TextLoader(str(DATA_FILE))
raw_docs = loader.load()

print("[Moni-AI] Embedding documents …")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def build_and_save_index():
    print("[Moni-AI] Building new FAISS index …")
    vs = FAISS.from_documents(raw_docs, embeddings)
    vs.save_local(INDEX_PATH)
    return vs

if args.rebuild_index or not INDEX_PATH.exists():
    vectorstore = build_and_save_index()
else:
    try:
        print("[Moni-AI] Loading existing FAISS index …")
        vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    except ValueError as e:
        print("[Moni-AI] Deserialization warning:", e)
        vectorstore = build_and_save_index()

retriever = vectorstore.as_retriever(search_type="similarity", k=4)

# ---------------------------------------------------------------------------
# INITIALISE LLM 
# ---------------------------------------------------------------------------
print(f"[Moni-AI] Connecting to Ollama at {OLLAMA_URL} with model '{OLLAMA_MODEL}' …")
try:
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE)
    # quick test call to ensure server/model is available
    _ = llm.invoke("ping")
    print("[Moni‑AI] ✅ Ollama backend ready.")
except Exception as err:
    print("[Moni‑AI] ⚠️  Ollama failed (", err, ")")
    if HF_MODEL_ID:
        print(f"[Moni‑AI] Falling back to HuggingFace model '{HF_MODEL_ID}'.")
        tokenizer = transformers.AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        generate_pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )
        llm = HuggingFacePipeline(pipeline=generate_pipeline)
    else:
        print("[Moni‑AI] No valid LLM backend available. Exiting.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Moni-AI, a clinical assistant.\n\n"
        "Context:\n{context}\n\n"
        "Patient input:\n{question}\n\n"
        "Return:\n"
        "1. **Primary diagnosis / classification**\n"
        "2. **Reasoning in 3-5 bullet points**\n"
        "3. **Up to three differential diagnoses** (if relevant)"
    ),
)

# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------
callbacks: List = []
if os.getenv("LANGSMITH_TRACING"):
    callbacks.append(LangChainTracer())
    print("[Moni-AI] LangSmith tracing enabled via LangChainTracer.")
else:
    callbacks.append(StdOutCallbackHandler())

qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template, callbacks=callbacks)
rag_chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever, callbacks=callbacks)

# ---------------------------------------------------------------------------
# CORE FUNCTION
# ---------------------------------------------------------------------------

def classify_case(text: str) -> str:
    """Run the RAG pipeline on input text and return the LLM answer."""
    return rag_chain.run(text)

# ---------------------------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------------------------

demo = gr.Interface(
    fn=classify_case,
    inputs=gr.Textbox(lines=12, label="Patient Case Notes or Symptoms"),
    outputs=gr.Textbox(label="AI Diagnosis & Reasoning"),
    title="Moni-AI: Psychiatric Assistant",
    description="Paste patient notes, symptoms, or assessments. Receive a concise classification, reasoning, and differentials.",
)

# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if args.cli:
        print("Moni-AI CLI – paste case notes (empty line to quit)\n")
        while True:
            user_input = input("> ")
            if not user_input.strip():
                break
            print("\n--- Moni-AI Response ---")
            print(classify_case(user_input))
            print("---------------------------\n")
    else:
        demo.launch()

