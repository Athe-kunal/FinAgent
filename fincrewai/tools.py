from src.vectorDatabaseDocker import create_database
from src.queryDatabase import query_database_earnings_call,query_database_sec
from functools import partial
from crewai_tools import tool

def build_database_and_query(ticker:str, year:str):
    (
        qdrant_client,
        encoder,
        speakers_list_1,
        speakers_list_2,
        speakers_list_3,
        speakers_list_4,
        sec_form_names,
        earnings_call_quarter_vals,
    ) = create_database(ticker=ticker, year=int(float(year)))
    query_sec = partial(query_database_sec,qdrant_client=qdrant_client,encoder=encoder)
    query_earnings_call = partial(query_database_earnings_call,qdrant_client=qdrant_client,encoder=encoder)

    sec_tools_list = []
    for sec_form in sec_form_names:
        if "10-K" in sec_form:
            # description = 
            @tool("10-K data")
            def yearly_tool(question:str)->str:
                """
                This tool will fetch relevant documents from 10-K yearly filing to answer questions regarding the financial and other conditions about a public company 
                """
                return query_sec(question=question,search_form=sec_form)
            # sec_tools.update({sec_form:yearly_tool})
            sec_tools_list.append(yearly_tool)
        elif "10-Q" in sec_form:
            quarter = sec_form.split("-")[1]
            @tool(f"{sec_form} data")
            def quarterly_tool(question:str)->str:
                "This tool will fetch relevant documents from 10-Q quarterly filing from the quarter mentioned in the name of the tool." 
                "This tool will answer questions regarding the quarterly financial and other conditions about a public company" 
                return query_sec(question=question,search_form=sec_form)
            # sec_tools.update({sec_form:quarterly_tool})
            sec_tools_list.append(quarterly_tool)
    earnings_calls_tools_list = []
    for idx,speaker_list in enumerate([speakers_list_1,speakers_list_2,speakers_list_3,speakers_list_4]):
        if speaker_list == []: continue
        else:
            quarter = earnings_call_quarter_vals[idx]
            @tool(f"{quarter} earnings call data")
            def earnings_call_tool(question:str)->str:
                """
                This tool will fetch relevant documents from earnings call transcripts from the quarter mentioned in the name of the tool
                to answer questions regarding the financial and other conditions about a public company 
                """
                return query_earnings_call(question=question,quarter=quarter,speakers_list=speaker_list)
            # sec_tools.update({sec_form:yearly_tool})
            earnings_calls_tools_list.append(earnings_call_tool)
    return sec_tools_list,earnings_calls_tools_list
