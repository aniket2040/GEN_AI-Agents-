from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import result

load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=5)

llm = ChatMistralAI(model_name="mistral-small-2506")

prompt = ChatPromptTemplate.from_template(
    """
    you are a helpfull assistant
    
    summarize the following news clear bullet points
    
    {news}
    """
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("latest AI news of 2026")

result = chain.invoke({"news": news_result})

print(result)