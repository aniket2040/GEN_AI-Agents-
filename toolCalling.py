from dotenv import load_dotenv
from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from sqlalchemy.ext.asyncio import result

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from rich import print

@tool
def get_text_length(text: str) -> int:
    return len(text)

llm = ChatMistralAI(model_name= "mistral-small-2506")

tools = {
    "get_text_length": get_text_length
}
# tool binding
llm_tool = llm.bind_tools([get_text_length])

message = []
prompt = input("How Are You")
query = HumanMessage(f"Return the number of given character in the given text : {prompt} ")
message.append(query)
print(message)

result = llm_tool.invoke(message)
message.append(result)
print(message)

if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)



result = llm_tool.invoke(message)
print(result.content)










# result2 = llm_tool.invoke("Return the number od character in the given text : 'Hello World'")
#
# if result2.tool_calls :
#     tool_call = result2.tool_calls[0]
#
#     tool_name = tool_call["name"]
#     tool_args = tool_call["args"]
#
#     tool_result = get_text_length.invoke(tool_args)
#
#     final_response = llm_tool.invoke(f"the length of text is {tool_result}")
#     print(final_response)