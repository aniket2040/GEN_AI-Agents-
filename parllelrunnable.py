from dotenv import load_dotenv
from httptools.parser import parser
from langchain_classic.chains.llm import LLMChain
from langgraph.channels import topic
from langchain_core.runnables import RunnableParallel, RunnableLambda
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in details"
)
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

model = ChatMistralAI(model_name = "mistral-small-2506")

topic = "Machine Learning"
parser = StrOutputParser()

chain = RunnableParallel({
    "short" : RunnableLambda(lambda x:x ['short']) | short_prompt | model | parser ,
    "detailed" : RunnableLambda(lambda x:x ['detailed']) | detailed_prompt | model | parser
})

result = chain.invoke({
    "short" : {"topic":"Machine Learning"},
    "detailed" : {"topic":"Deep Learning"}
})
print(result['short'])
print(result['detailed'])
# short_prompt | model | parser
# detailed_prompt | model | parser
# parser = StrOutputParser()

# chain = promt | model | parser
#
# result = chain.invoke("Machine Learning")
#
# print(result)



