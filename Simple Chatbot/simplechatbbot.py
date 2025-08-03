#import the packages
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import streamlit as st

#setup api key to use Gemini API
from dotenv import load_dotenv
load_dotenv()
apiKey = os.getenv('GOOGLE_API_KEY')

#setup a LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash" ,api_key = apiKey)

# query = input('Ask Query: ')
# res = llm.invoke(query)
# print(res.content)

#build the streamlit app
st.title('Personal AI Bot')
st.header('Ask me anything')
user_input = st.text_input("enter query", key='input')
if user_input:
    res = llm.invoke(user_input)
    st.write(res.content)


