import sys
import subprocess
import os

print("🔧 STARTING ENVIRONMENT REPAIR...")
print(f"🐍 Python Executable: {sys.executable}")

def install_package(package):
    print(f"📦 Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--force-reinstall"])

def uninstall_package(package):
    print(f"🗑️ Uninstalling {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])

try:
    # 1. Inspect the broken package
    import qdrant_client
    print(f"📍 Current Qdrant Location: {os.path.dirname(qdrant_client.__file__)}")
    
    if hasattr(qdrant_client, '__version__'):
        print(f"ℹ️  Current Version: {qdrant_client.__version__}")
    else:
        print("⚠️  WARNING: Package has no version! It is likely corrupted or shadowed.")

except ImportError:
    print("❌ Qdrant not found (Clean slate).")
except Exception as e:
    print(f"⚠️  Error during inspection: {e}")

# 2. THE FIX: Nuke and Reinstall
print("\n🚀 EXECUTING CLEAN RE-INSTALL...")
try:
    uninstall_package("qdrant-client")
    uninstall_package("qdrant-client") # Twice to be sure
    
    # Install specific stable version
    install_package("qdrant-client==1.9.0")
    
    print("\n✅ RE-INSTALL COMPLETE.")
    
    # 3. Verify
    import qdrant_client
    from qdrant_client import QdrantClient
    print(f"🎉 Success! New Location: {os.path.dirname(qdrant_client.__file__)}")
    print(f"🎉 New Version: {qdrant_client.__version__}")
    
    # Check for the search method
    client = QdrantClient(location=":memory:")
    if hasattr(client, 'search'):
        print("✅ Method 'search' VERIFIED. You are ready.")
    else:
        print("❌ Method 'search' still missing. Something is very strange.")

except Exception as e:
    print(f"\n❌ FATAL ERROR DURING FIX: {e}")
    
print("\n👉 Please try running 'python main_agent_pro_2.py' now.")