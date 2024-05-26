from crewai import Task

def get_tasks(sec_tools_list,sec_filings_agent,earnings_calls_tools_list,earnings_call_transcripts_agent,books_tools,books_agent):
    sec_task = Task(
        description = (
            "Use one of the tool to get the relevant documents to the question {question}."
            "Using the relevant context only, answer in detail to the question, and don't skip out on numerical figures (if any)."
        ),
        expected_output = (
            "Detailed answer to the question, including the relevant context and numerical figures"
        ),
        tools = sec_tools_list,
        agent = sec_filings_agent
    )

    earnings_call_task = Task(
        description = (
            "Use one of the tool to get the relevant documents to the question {question}."
            "Using the relevant context only, answer in detail to the question, and don't skip out on numerical figures (if any)."
            "Also, mention the speakers from relevant context"
        ),
        expected_output = (
            "Detailed answer to the question, including the relevant context and numerical figures"
        ),
        tools = earnings_calls_tools_list,
        agent = earnings_call_transcripts_agent
    )
    books_task = Task(
        description = (
            "Use the tool to get relevant documents to the question {question} which is about investing philosophies and valuation of companies."
            "Using the relevant context only, answer in detail to the question."
        ),
        expected_output = (
            "Detailed answer to the question, including the relevant context and numerical figures"
        ),
        tools = books_tools,
        agent = books_agent)
    return sec_task, earnings_call_task, books_task