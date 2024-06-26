from crewai import Agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def get_agents(sec_tools_list, earnings_calls_tools_list, books_tool):
    sec_filings_agent = Agent(
        role="SEC Filings Financial Data Analyst",
        goal="Answer the question in detail by first using the relevant tool to get relevant documents ",
        backstory="With access to 10-K yearly SEC filings and 10-Q quarterly SEC filings "
        "you can get relevant documents from these documents to answer the question. "
        "Don't answer the questions which is outside the scope of the relevant documents. "
        "Include all the relevant numerical figures in your answer.",
        verbose=True,
        allow_delegation=False,
        tools=sec_tools_list,
        llm=ChatOpenAI(model="gpt-3.5-turbo-0125"),
    )

    earnings_call_transcripts_agent = Agent(
        role="Earnings Call transcripts Financial Data Analyst",
        goal="Answer the question in detail by first using the relevant tool from the relevant quarter to get relevant documents. ",
        backstory="With access to earnings call transcripts quarterly data you can get relevant documents from these documents to answer the question. "
        "The earnings call transcripts has data about what the executive team of the company said about the company current financials and future directions. "
        "Don't answer the questions which is outside the scope of the relevant documents. "
        "Include all the relevant numerical figures in your answer.",
        verbose=True,
        allow_delegation=False,
        tools=earnings_calls_tools_list,
        llm=ChatOpenAI(model="gpt-3.5-turbo-0125"),
    )

    books_agents = Agent(
        role="Financial and Valuation Expert",
        goal="Answer the question in detail by first using the relevant tool to get relevant documents. ",
        backstory="With access to financial and valuation books written by industry experts on the philosophy of investment, pitfalls of investing and how to value a company, "
        "you can get relevant documents from this agent to answer the question. Questions about how to invest, applied corporate finance, risk management in investment "
        "can be answered using this agent.",
        verbose=True,
        allow_delegation=False,
        tools=books_tool,
        llm=ChatOpenAI(model="gpt-3.5-turbo-0125"),
    )
    return sec_filings_agent, earnings_call_transcripts_agent, books_agents
