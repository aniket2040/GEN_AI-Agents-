from dotenv import load_dotenv
from httptools.parser import parser
from langchain_classic.chains.llm import LLMChain
from langgraph.channels import topic

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

promt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

model = ChatMistralAI(model_name = "mistral-small-2506")

parser = StrOutputParser()

chain = promt | model | parser

result = chain.invoke("Machine Learning")

print(result)



