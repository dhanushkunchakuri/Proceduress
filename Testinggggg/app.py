import os
import re
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# NEW proper Ollama package (no deprecation warnings)
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# -------------------------
# Helpers: parsing + graph
# -------------------------
PROC_NAME_RE = re.compile(r"CREATE\s+PROCEDURE\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
EXEC_RE = re.compile(r"EXEC\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
TABLE_RE = re.compile(r"(FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

def parse_sql_procedures(sql_text: str):
    """Split a large SQL text into {proc_name: proc_sql} using CREATE PROCEDURE boundaries."""
    matches = list(PROC_NAME_RE.finditer(sql_text))
    procedures = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(sql_text)
        proc_sql = sql_text[start:end].strip()
        procedures[name] = proc_sql
    return procedures

def extract_called_procs(sql: str):
    return EXEC_RE.findall(sql)

def extract_tables(sql: str):
    return [tbl for _, tbl in TABLE_RE.findall(sql)]

def build_calls_and_tables_map(procs: dict):
    calls = {}
    tables = {}
    for name, sql in procs.items():
        calls[name] = extract_called_procs(sql)
        tables[name] = extract_tables(sql)
    return calls, tables

def resolve_tables(proc_name: str, calls_map: dict, tables_map: dict, visited=None):
    """Recursively collect all tables used by proc_name and its callee graph."""
    if visited is None:
        visited = set()
    if proc_name in visited:
        return []
    visited.add(proc_name)
    out = list(tables_map.get(proc_name, []))
    for callee in calls_map.get(proc_name, []):
        out.extend(resolve_tables(callee, calls_map, tables_map, visited))
    return list(dict.fromkeys(out))  # preserve order, unique

def expand_proc_sql(proc_name: str, procs: dict, calls_map: dict, visited=None):
    """Return the proc SQL plus the SQL of all callee procs recursively (expanded components)."""
    if visited is None:
        visited = set()
    if proc_name in visited:
        return ""  # avoid cycles
    visited.add(proc_name)
    base = procs.get(proc_name, "")
    parts = [base]
    for callee in calls_map.get(proc_name, []):
        parts.append("\n\n-- Expanded from: " + callee + "\n" + expand_proc_sql(callee, procs, calls_map, visited))
    return "\n".join(parts)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="SQL Procedures → RAG with Ollama", layout="wide")
st.title("RAG ingestion for SQL procedures (Ollama + FAISS)")

sql_input = st.text_area("Paste SQL (many procedures) here", height=360)
query = st.text_input("Ask a question about your procedures (e.g. 'Which tables does GetFullCustomerProfile use?')")

# Option to persist vector DB to disk
persist_path = st.text_input("Optional: path to save/load FAISS index (leave blank for in-memory)", value="")

if st.button("Index and Query") and sql_input and query:
    with st.spinner("Parsing procedures..."):
        procs = parse_sql_procedures(sql_input)
        if not procs:
            st.error("No procedures found. Make sure your SQL contains CREATE PROCEDURE statements.")
            st.stop()

        calls_map, tables_map = build_calls_and_tables_map(procs)

    with st.spinner("Building enriched documents and expanding components..."):
        enriched_texts = []
        for name in procs:
            expanded_sql = expand_proc_sql(name, procs, calls_map)
            resolved = resolve_tables(name, calls_map, tables_map)
            called = calls_map.get(name, [])
            enriched = (
                f"Procedure Name: {name}\n"
                f"Procedures Called: {called}\n"
                f"Resolved Tables (direct + indirect): {resolved}\n\n"
                f"SQL:\n{expanded_sql}"
            )
            enriched_texts.append((name, enriched))

    # split into chunks but keep the procedure name in each chunk so metadata can be inferred
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = []
    for name, text in enriched_texts:
        chunks = splitter.create_documents([text])  # returns list[Document]
        # attach metadata (procedure name + resolved tables)
        for ch in chunks:
            # try to extract the Resolved Tables line (we included it above)
            m = re.search(r"Resolved Tables \(direct \+ indirect\):\s*(\[[^\]]*\])", ch.page_content)
            tables_meta = []
            if m:
                try:
                    tables_meta = eval(m.group(1))
                except Exception:
                    tables_meta = []
            ch.metadata = {"proc_name": name, "tables": tables_meta}
            docs.append(ch)

    # embeddings + FAISS
    with st.spinner("Indexing embeddings into FAISS (this may take a moment)..."):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")  # ensure pulled
        if persist_path:
            # load if exists, otherwise create and save
            if os.path.exists(persist_path):
                vector_db = FAISS.load_local(persist_path, embeddings)
            else:
                vector_db = FAISS.from_documents(docs, embeddings)
                vector_db.save_local(persist_path)
        else:
            vector_db = FAISS.from_documents(docs, embeddings)

        # better to retrieve more to ensure all components considered
        retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # show graph summary on the left
    st.subheader("Procedures parsed & dependency summary")
    for pname in procs:
        st.write(f"- **{pname}** calls: `{calls_map.get(pname, [])}` tables: `{tables_map.get(pname, [])}`")

    # retrieve relevant docs for the user's query (so you can inspect what RAG pulled)
    with st.spinner("Retriever: fetching relevant components..."):
        retrieved_docs = retriever.get_relevant_documents(query)

    st.subheader("RAG: Retrieved documents (top results)")
    for i, d in enumerate(retrieved_docs, start=1):
        proc_meta = d.metadata.get("proc_name", "unknown")
        st.markdown(f"**{i}. Procedure:** `{proc_meta}` — tables: `{d.metadata.get('tables', [])}`")
        # show a short excerpt
        excerpt = d.page_content[:800].replace("\n", " ")
        st.code(excerpt + ("..." if len(d.page_content) > 800 else ""))

    # Now run the QA chain (retriever + LLM)
    st.subheader("Ollama (gemma:2b) answer")
    with st.spinner("Asking the LLM..."):
        llm = OllamaLLM(model="gemma:2b")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
        # Use invoke (newer API) with dict query
        result = qa_chain.invoke({"query": query})
        answer = result.get("result") if isinstance(result, dict) else str(result)

    st.success("Answer:")
    st.write(answer)