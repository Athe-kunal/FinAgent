from src.utils import get_earnings_transcript
import re
from langchain.schema import Document
from src.config import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from tqdm.notebook import tqdm
from src.secData import sec_main
from tenacity import RetryError
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from dotenv import load_dotenv,find_dotenv
from langchain_chroma import Chroma
import os

def clean_speakers(speaker):
    speaker = re.sub("\n", "", speaker)
    speaker = re.sub(":", "", speaker)
    return speaker


def get_earnings_all_quarters_data(docs, quarter: str, ticker: str, year: int):
    resp_dict = get_earnings_transcript(quarter, ticker, year)

    content = resp_dict["content"]
    pattern = re.compile(r"\n(.*?):")
    matches = pattern.finditer(content)

    speakers_list = []
    ranges = []
    for match_ in matches:
        # print(match.span())
        span_range = match_.span()
        # first_idx = span_range[0]
        # last_idx = span_range[1]
        ranges.append(span_range)
        speakers_list.append(match_.group())
    speakers_list = [clean_speakers(sl) for sl in speakers_list]

    for idx, speaker in enumerate(speakers_list[:-1]):
        start_range = ranges[idx][1]
        end_range = ranges[idx + 1][0]
        speaker_text = content[start_range + 1 : end_range]

        docs.append(
            Document(
                page_content=speaker_text,
                metadata={"speaker": speaker, "quarter": quarter},
            )
        )

    docs.append(
        Document(
            page_content=content[ranges[-1][1] :],
            metadata={"speaker": speakers_list[-1], "quarter": quarter},
        )
    )
    return docs, speakers_list


def get_all_docs(ticker: str, year: int):
    docs = []
    earnings_call_quarter_vals = []
    print("Earnings Call Q1")
    try:
        docs, speakers_list_1 = get_earnings_all_quarters_data(docs, "Q1", ticker, year)
        earnings_call_quarter_vals.append("Q1")
    except RetryError:
        print(f"Don't have the data for Q1")
        speakers_list_1 = []

    print("Earnings Call Q2")
    try:
        docs, speakers_list_2 = get_earnings_all_quarters_data(docs, "Q2", ticker, year)
        earnings_call_quarter_vals.append("Q2")
    except RetryError:
        print(f"Don't have the data for Q2")
        speakers_list_2 = []
    print("Earnings Call Q3")
    try:
        docs, speakers_list_3 = get_earnings_all_quarters_data(docs, "Q3", ticker, year)
        earnings_call_quarter_vals.append("Q3")
    except RetryError:
        print(f"Don't have the data for Q3")
        speakers_list_3 = []
    print("Earnings Call Q4")
    try:
        docs, speakers_list_4 = get_earnings_all_quarters_data(docs, "Q4", ticker, year)
        earnings_call_quarter_vals.append("Q4")
    except RetryError:
        print(f"Don't have the data for Q4")
        speakers_list_4 = []
    print("SEC")
    section_texts, sec_form_names = sec_main(ticker, year)

    for filings in section_texts:
        texts_dict = filings[-1]

        for section_name, text in texts_dict.items():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "accessionNumber": filings[0],
                        "filing_type": filings[1],
                        "filingDate": filings[2],
                        "reportDate": filings[3],
                        "sectionName": section_name,
                    },
                )
            )
    return (
        docs,
        sec_form_names,
        earnings_call_quarter_vals,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
    )

def create_database(ticker: str, year: int,embed_type:str="openai",db_path:str="./sec_calls_chromadb"):
    """Build the database to query from it

    Args:
        quarter (str): The quarter of the earnings call
        ticker (str): The ticker of the company
        year (int): The year of the earnings call
    """
    collection_name = f"{ticker}_{year}_{embed_type}"
    (
        docs,
        sec_form_names,
        earnings_call_quarter_vals,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
    ) = get_all_docs(ticker=ticker, year=year)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # is_separator_regex = False,
    )
    split_docs = text_splitter.split_documents(docs)
    if embed_type == "openai":
        load_dotenv(find_dotenv(),override=True)
        emb_fn = OpenAIEmbeddings(model="text-embedding-3-small",api_key=os.environ['OPENAI_API_KEY'])
    elif embed_type == "sentence_transformer":
        emb_fn = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(os.path.join(db_path,collection_name)):
        try:
            chromadb = Chroma(persist_directory=db_path,embedding_function=emb_fn,collection_name=collection_name)
        except:
            chromadb = Chroma.from_documents(split_docs, emb_fn, persist_directory=db_path)
    else:
        chromadb = Chroma.from_documents(split_docs, emb_fn, persist_directory=db_path)

    return (
        chromadb,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
        sec_form_names,
        earnings_call_quarter_vals,
    )