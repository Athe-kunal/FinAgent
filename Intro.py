import streamlit as st
from datetime import datetime
from main import get_crewai_agent

curr_year = datetime.now().year
ticker = st.text_input(label="Ticker")
year = st.text_input(label="Year")

if year != "":
    int_year = int(float(year))
submit_button = st.button(label="Submit")
if ticker != "" and year != "" and submit_button:
    if curr_year == int_year:
        curr_year_bool = True
    else:
        curr_year_bool = False
    finance_docs_crewai = get_crewai_agent(ticker, year)
    st.session_state["ticker"] = ticker
    st.session_state["year"] = str(year)
    st.session_state["finance_docs_crew"] = finance_docs_crewai
    st.write("Created Database")
