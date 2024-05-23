from fincrewai.agents import get_agents
from fincrewai.tasks import get_tasks
from fincrewai.tools import build_database_and_query
from fincrewai.crew import get_crew

def get_crewai_agent(ticker:str,year:str):
    """Get the crewai agent

    Args:
        ticker (str): The ticker of the company
        year (str): The year of the earnings call
    """
    sec_tools_list,earnings_calls_tools_list = build_database_and_query(ticker=ticker,year=year)
    sec_filings_agent, earnings_call_transcripts_agent = get_agents(sec_tools_list=sec_tools_list,earnings_calls_tools_list=earnings_calls_tools_list)
    sec_task, earnings_call_task = get_tasks(sec_tools_list,sec_filings_agent,earnings_calls_tools_list,earnings_call_transcripts_agent)
    financial_docs_crew = get_crew(sec_filings_agent,earnings_call_transcripts_agent,sec_task,earnings_call_task)
    return financial_docs_crew