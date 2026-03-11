import pandas as pd
from pymongo import MongoClient

df = pd.read_csv("fraud.csv")

client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_db"]
collection = db["transactions"]

collection.insert_many(df.to_dict("records"))

print("Simple fraud data stored in MongoDB")

#view the data
for transaction in collection.find():
    print(transaction)
    