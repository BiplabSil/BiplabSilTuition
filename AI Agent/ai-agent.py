#setup the tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
tool = TavilySearchResults(k=3)
tools = [tool]

#setup LLM
from dotenv import load_dotenv
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#setup Prompt template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, response to user query from the available tool."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
#setup agent
from langchain.agents import create_tool_calling_agent, AgentExecutor
agent = create_tool_calling_agent(llm=llm,tools=tools,prompt=prompt)

agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

#res = agent_executor.invoke({"input":"who is the winner of norway chess 2025?"})
#print(res)
#setup streamlit
import streamlit as st
st.title("Personal AI Agent")
st.header("Enter your query below")
user_input = st.text_input("", key="input")
if user_input:
    response = agent_executor.invoke({"input":user_input})
    st.write(response['output'])