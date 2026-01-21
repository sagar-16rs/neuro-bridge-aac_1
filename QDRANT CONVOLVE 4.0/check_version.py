import qdrant_client
print(f"✅ Qdrant Client Version: {qdrant_client.__version__}")

try:
    client = qdrant_client.QdrantClient(path="test_db")
    print("✅ Client initialized successfully.")
    if hasattr(client, 'search'):
        print("✅ 'search' method FOUND.")
    else:
        print("❌ 'search' method NOT found.")
except Exception as e:
    print(f"❌ Error: {e}")