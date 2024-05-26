import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


def build_database():
    book_dir = "BOOKS_MD"

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, return_each_line=False
    )
    all_docs = []
    for book_name in os.listdir(book_dir):
        books_path = os.path.join(book_dir, book_name)
        with open(f"{books_path}/{book_name}.md", "r", encoding="utf-8") as input_file:
            text = input_file.read()
        md_header_splits = markdown_splitter.split_text(text)
        # print(md_header_splits)
        for md in md_header_splits:
            md.metadata.update({"book_name": book_name})
        # break
        all_docs.extend(md_header_splits)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chroma_db = Chroma.from_documents(
        all_docs, embedding_function, persist_directory="./chroma_db"
    )

    return chroma_db


def load_database():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    return db3
