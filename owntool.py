import langchain.tools import tool

@tool
def get_greeting(name : str) -> str:
    """
     Generate a greeting message for the user
    :param name:
    :return:
    """
    return f"Hello, {name} , Welcome to AI world!"

result = get_greeting.invoke({"name":"Yash"})
result2 = get_greeting.invoke({"name":"Paras"})

print(result)
print(result2)
print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)