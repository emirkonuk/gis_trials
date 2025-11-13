import time
import sys
import warnings
from qdrant_client import QdrantClient

# --- Config ---
COLLECTION_NAME = "hemnet_listings_v1"
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
RETRIES = 10
RETRY_DELAY = 3
TEXT_DIM = 384
IMAGE_DIM = 768

def main():
    """
    Connects to Qdrant with retries and recreates the collection.
    """
    # Suppress the "recreate_collection is deprecated" warning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    print(f"--- Purge Qdrant Script ---")
    print(f"Attempting to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    
    client = None
    
    # --- 1. Connect to Qdrant with a retry loop ---
    for i in range(RETRIES):
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            # Try a simple health check
            client.health_check()
            print("Successfully connected to Qdrant.")
            break
        except Exception as e:
            print(f"Attempt {i+1}/{RETRIES}: Qdrant not ready, retrying in {RETRY_DELAY}s... (Error: {e})")
            time.sleep(RETRY_DELAY)
            
    if client is None:
        print(f"Error: Could not connect to Qdrant after {RETRIES * RETRY_DELAY} seconds.")
        sys.exit(1)

    # --- 2. Delete the old collection (if it exists) ---
    try:
        print(f"Deleting collection {COLLECTION_NAME}...")
        client.delete_collection(COLLECTION_NAME)
        print(f"Successfully deleted collection {COLLECTION_NAME}.")
    except Exception as e:
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            print(f"Collection {COLLECTION_NAME} does not exist, continuing.")
        else:
            print(f"An unexpected error occurred while deleting: {e}")
            # Continue anyway, as recreate_collection will handle it

    # --- 3. Recreate the collection ---
    try:
        print(f"Recreating collection {COLLECTION_NAME}...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "text": {"size": TEXT_DIM, "distance": "Cosine"},
                "image": {"size": IMAGE_DIM, "distance": "Cosine"}
            }
        )
        print("Qdrant collection purged and recreated successfully.")
        print(f"Text dim: {TEXT_DIM}, Image dim: {IMAGE_DIM}")
    except Exception as e:
        print(f"Error: Failed to recreate collection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()