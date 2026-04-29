from pathlib import Path

# =========================
# BASE PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Shared database (all users)
DB_PATH = DATA_DIR / "rag_database.db"

# OAuth credentials file
CREDENTIALS_FILE = BASE_DIR / "credentials.json"


# =========================
# PER-USER DYNAMIC PATHS
# =========================
def get_user_paths(user_id):
    """
    Return a dict of user-specific file paths.
    Directories are created automatically.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "chunks_path":   DATA_DIR / f"user_{user_id}_chunks.json",
        "metadata_path": DATA_DIR / f"user_{user_id}_metadata.json",
        "index_path":    MODELS_DIR / f"user_{user_id}_index.faiss",
        "sync_file":     DATA_DIR / f"user_{user_id}_sync.json",
    }