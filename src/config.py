from pathlib import Path
 
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
 
DB_PATH = BASE_DIR / "data" / "rag_database.db"
CHUNKS_PATH = BASE_DIR / "data" / "chunks.json"
METADATA_PATH = BASE_DIR / "data" / "metadata.json"
INDEX_PATH = BASE_DIR / "models" / "email_index.faiss"