from lightning_sdk import Studio

s = Studio(name="michi-adapter-train", teamspace="llm-development-project")
print("Final Studio Status:", s.status)
