import matplotlib.pyplot as plt
from qdrant_client import QdrantClient

# Connect to the DB
client = QdrantClient(path="qdrant_db_v9")

# Fetch Logs
logs, _ = client.scroll(
    collection_name="interaction_logs",
    limit=1000,
    with_payload=True
)

# Extract Data for Plotting
timestamps = []
confidences = []
fatigue_levels = []
actions = []

start_time = logs[0].payload['timestamp']

for point in logs:
    p = point.payload
    # Normalize time to start at 0
    timestamps.append(p['timestamp'] - start_time)
    confidences.append(p['confidence'])
    fatigue_levels.append(p['fatigue_level'])
    actions.append(p['action_taken'])

# --- GENERATE RESEARCH GRAPH ---
plt.figure(figsize=(10, 6))

# Plot 1: Neural Confidence vs Time
plt.plot(timestamps, confidences, label="Neural Confidence", color='blue', alpha=0.6)

# Plot 2: Fatigue Level
plt.plot(timestamps, fatigue_levels, label="Patient Fatigue", color='red', linestyle='--')

# Plot 3: Active Learning Moments
# Mark where the Agent triggered "ASK_CONFIRMATION"
ask_times = [t for t, a in zip(timestamps, actions) if a == "ASK_CONFIRMATION"]
ask_vals = [c for c, a in zip(confidences, actions) if a == "ASK_CONFIRMATION"]
plt.scatter(ask_times, ask_vals, color='orange', s=100, label="Agent Negotiation", zorder=5)

# Mark where the Agent "LEARNED"
learn_times = [t for t, a in zip(timestamps, actions) if a == "LEARN_AND_EXECUTE"]
learn_vals = [c for c, a in zip(confidences, actions) if a == "LEARN_AND_EXECUTE"]
plt.scatter(learn_times, learn_vals, color='green', marker='*', s=200, label="Memory Evolution", zorder=5)

plt.title("Neuro-Bridge 9.0: Active Learning & Fatigue Adaptation")
plt.xlabel("Session Time (seconds)")
plt.ylabel("Confidence / Fatigue Score")
plt.axhline(y=0.85, color='gray', linestyle=':', label="Standard Threshold")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("research_graph.png")
print("✅ Research Graph Generated: research_graph.png")