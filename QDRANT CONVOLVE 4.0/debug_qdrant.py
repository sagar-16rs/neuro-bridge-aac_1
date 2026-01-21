import qdrant_client
import os

print("\n--- DETECTIVE MODE ---")
print(f"1. Python thinks 'qdrant_client' is located at:")
print(f"   {qdrant_client.__file__}")

from qdrant_client import QdrantClient
try:
    c = QdrantClient(path="./debug_mem")
    print(f"\n2. Testing 'search' capability...")
    if hasattr(c, 'search'):
        print("   ✅ SUCCESS: .search() method exists!")
    else:
        print("   ❌ FAILURE: .search() is MISSING.")
        print("   Here are the methods this object actually has:")
        print("   ", [m for m in dir(c) if not m.startswith('_')])
except Exception as e:
    print(f"   ❌ Error initializing: {e}")

print("----------------------\n")