import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri = os.getenv("MONGODB_URI")
if not uri:
    print("No MONGODB_URI in .env")
    exit(1)
client = MongoClient(uri)
try:
    client.admin.command("ping")
    print("Connected successfully!")
except Exception as e:
    print("Connection failed:", e)