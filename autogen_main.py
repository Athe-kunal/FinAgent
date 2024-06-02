import streamlit as st
from datetime import datetime
from data_source.vectorDatabaseChroma import create_database
from books_db import load_database
from autogen import ConversableAgent
from dotenv import load_dotenv,find_dotenv
from autogen import register_function

load_dotenv(find_dotenv(),override=True)

curr_year = datetime.now().year
ticker = st.text_input(label="Ticker")
year = st.text_input(label="Year")

if year != "":
    int_year = int(float(year))
submit_button = st.button(label="Submit")
if ticker != "" and year != "" and submit_button:
    st.session_state["ticker"] = ticker
    st.session_state["year"] = str(year)
    if curr_year == int_year:
        curr_year_bool = True
    else:
        curr_year_bool = False
    
    langchain_chromadb,speakers_list_1,speakers_list_2,speakers_list_3,speakers_list_4,sec_form_names,earnings_call_quarter_vals = create_database(ticker,year,embed_type="sentence_transformer")
    quarter_speaker_dict = {
        "Q1":speakers_list_1,
        "Q2":speakers_list_2,
        "Q3":speakers_list_3,
        "Q4":speakers_list_4
    }
    def query_database_earnings_call(
        question: str,
        quarter: str
    )->str:
        """This tool will query the earnings call transcripts database for a given question and quarter and it will retrieve
        the relevant text along from the earnings call and the speaker who addressed the relevant documents. This tool helps in answering questions
        from the earnings call transcripts.

        Args:
            question (str): _description_. Question to query the database for relevant documents.
            quarter (str): _description_. the financial quarter that is discussed in the question and possible options are Q1, Q2, Q3, Q4

        Returns:
            str: relevant text along from the earnings call and the speaker who addressed the relevant documents
        """
        assert quarter in earnings_call_quarter_vals, "The quarter should be from Q1, Q2, Q3, Q4"
        
        req_speaker_list = []
        quarter_speaker_list = quarter_speaker_dict[quarter]

        for sl in quarter_speaker_list:
            if sl in question or sl.lower() in question:
                req_speaker_list.append(sl)
        if len(req_speaker_list) == 0:
            req_speaker_list = quarter_speaker_list
        
        relevant_docs = langchain_chromadb.similarity_search(
            question,
            k=5,
            filter={
                "$and":[
                    {
                        "quarter":{"$eq":quarter}
                    },
                    {
                        "speaker":{"$in":req_speaker_list}
                    }
                ]
            }
        )

        speaker_releavnt_dict = {}
        for doc in relevant_docs:
            speaker = doc.metadata['speaker']
            speaker_text = doc.page_content
            if speaker not in speaker_releavnt_dict:
                speaker_releavnt_dict[speaker] = speaker_text
            else:
                speaker_releavnt_dict[speaker] += " "+speaker_text
        
        relevant_speaker_text = ""
        for speaker, text in speaker_releavnt_dict.items():
            relevant_speaker_text += speaker + ": "
            relevant_speaker_text += text + "\n\n"

        return relevant_speaker_text

    def query_database_sec(
            question: str,
            sec_form_name: str
    )->str:
        """This tool will query the SEC Filings database for a given question and form name, and it will retrieve
        the relevant text along from the SEC filings and the section names. This tool helps in answering questions
        from the sec filings.

        Args:
            question (str): _description_. Question to query the database for relevant documents
            sec_form_name (str): _description_. SEC FORM NAME that the question is talking about. It can be 10-K for yearly data and 10-Q for quarterly data. For quarterly data, it can be 10-Q2 to represent Quarter 2 and similarly for other quarters.

        Returns:
            str: Relevant context for the question from the sec filings
        """
        assert sec_form_name in sec_form_names, f'The search form type should be in {sec_form_names}'
        
        relevant_docs = langchain_chromadb.similarity_search(
            question,
            k=5,
            filter={
                "filing_type":{"$eq":sec_form_name}
            }
        )

        relevant_section_dict = {}
        for doc in relevant_docs:
            section = doc.metadata['sectionName']
            section_text = doc.page_content
            if section not in relevant_section_dict:
                relevant_section_dict[section] = section_text
            else:
                relevant_section_dict[section] += " "+section_text
        
        relevant_section_text = ""
        for section, text in relevant_section_dict.items():
            relevant_section_text += section + ": "
            relevant_section_text += text + "\n\n"

        return relevant_section_text
    books_collection = load_database()

    def query_database_books(
            question: str
    )->str:
        """This tool will query the financial books database for a given question and it will retrieve the relevant text.
            This tool helps in answering questions from the earnings call transcripts.

        Args:
            question (str): _description_. Question to query the database for relevant documents

        Returns:
            str: relevant documents from the financial books database
        """
        relevant_docs = books_collection.similarity_search(
            question,
            k=5,
        )
        releavnt_text = ""
        for doc in relevant_docs:
            releavnt_text += doc.page_content
            releavnt_text += "\n\n"
        return releavnt_text

    sec_form_system_msg = ""
    llm_config = {"model":"gpt-4-turbo"}
    for sec_form in sec_form_names:
        if sec_form == "10-K":
            sec_form_system_msg+= "10-K for yearly data, "
        elif "10-Q" in sec_form:
            quarter = sec_form[-1]
            sec_form_system_msg+= f"{sec_form} for Q{quarter} data, "
    sec_form_system_msg = sec_form_system_msg[:-2]

    earnings_call_system_message = ", ".join(earnings_call_quarter_vals)
    system_msg = f"""You are a helpful financial assistant and your task is to select the sec_filings or earnings_call or financial_books to best answer the question. 
    You can use query_database_sec(question,sec_form) by passing question and relevant sec_form names like {sec_form_system_msg} 
    or you can use query_database_earnings_call(question,quarter) by passing question and relevant quarter names with possible values {earnings_call_system_message} 
    or you can use query_database_books(question) to get relevant documents from financial textbooks about valuation and investing philosophies. When you are ready to end the coversation, reply TERMINATE"""


    user_proxy = ConversableAgent(
    name = "Planner Admin",
    system_message=system_msg,
    code_execution_config=False,
    llm_config=llm_config,
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
)
    tool_proxy = ConversableAgent(
    name="Tool Proxy",
    system_message="Analyze the response from user proxy and decide whether the suggested database is suitable "
    ". Answer in simple yes or no",
    llm_config=False,
    # is_termination_msg=lambda msg: "exit" in msg.get("content",""),
    default_auto_reply="Please select the right database.",
    human_input_mode="ALWAYS",
)
    
    
    tools_dict = {
        "sec":[query_database_sec,"Tool to query SEC filings database"],
        "earnings_call": [query_database_earnings_call, "Tool to query earnings call transcripts database"],
        "books": [query_database_books, "Tool to query books database"], 
    }

    for tool_name,tool in tools_dict.items():
        register_function(
            tool[0],
            caller=user_proxy,
            executor=tool_proxy,
            name = tool[0].__name__,
            description=tool[1]       
        )
    
    st.session_state['user_proxy'] = user_proxy
    st.session_state['tool_proxy'] = tool_proxy