import sqlite3
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

conn = sqlite3.connect("fpl_intel.db")

print("Loading news articles...")
news_df = pd.read_sql_query(
    "SELECT id, title, body, source FROM news_articles", conn
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

news_docs = []

for _, row in news_df.iterrows():
    full_text = f"Source: {row['source']}\nTitle: {row['title']}\nContent: {row['body']}"
    chunks = text_splitter.split_text(full_text)

    for chunk in chunks:
        news_docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "type": "news",
                    "article_id": row["id"],
                    "source": row["source"]
                }
            )
        )

print(f"Created {len(news_docs)} news chunks")

print("Loading players data...")
players_df = pd.read_sql_query("""
SELECT web_name, total_points, now_cost,
       goals_scored, assists, clean_sheets,
       form, news
FROM players
""", conn)

player_docs = []

for _, row in players_df.iterrows():
    content = f"Player: {row['web_name']}\nTotal Points: {row['total_points']}\nCost: {row['now_cost']/10}m\nGoals: {row['goals_scored']}\nAssists: {row['assists']}\nClean Sheets: {row['clean_sheets']}\nForm: {row['form']}\nNews: {row['news']}"

    player_docs.append(
        Document(
            page_content=content,
            metadata={
                "type": "player",
                "name": row["web_name"]
            }
        )
    )

print(f"Processed {len(player_docs)} players")

all_docs = news_docs + player_docs

print("Building vector database...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=all_docs,
    embedding=embeddings,
    persist_directory="./fpl_vector_db",
    collection_name="fpl_rag"
)

print("Done")
conn.close()