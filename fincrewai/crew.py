from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Define the crew with agents and tasks
def get_crew(sec_filings_agent,earnings_call_transcripts_agent,sec_task,earnings_call_task):
    financial_docs_crew = Crew(
        agents=[sec_filings_agent, 
                earnings_call_transcripts_agent],
        
        tasks=[sec_task, 
            earnings_call_task],
        
        manager_llm=ChatOpenAI(model="gpt-3.5-turbo", 
                            temperature=0.0),
        process=Process.hierarchical,
        verbose=True
        )
    return financial_docs_crew