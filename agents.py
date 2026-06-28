import os
import requests
from dotenv import load_dotenv
from rich import print
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# -------------------------------
# Weather Tool
# -------------------------------
@tool()
def get_weather(city: str) -> str:
    """
    Fetch weather data for a given city using OpenWeather API.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    print("[bold blue]Weather data:[/]", data)

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not retrieve data')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {temp}°C, {desc}"


# -------------------------------
# News Tool
# -------------------------------
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool()
def get_news(city: str) -> str:
    """
    Fetch latest news about a given city using Tavily API.
    """
    response = tavily_client.search(
        query=f"latest news about {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])
    if not results:
        return f"Error: No news found for {city}"

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(f"- {title}\n{url}\n{snippet[:100]}...")

    return f"Latest news about {city}:\n\n" + "\n\n".join(news_list)


# -------------------------------
# LLM Setup
# -------------------------------
llm = ChatMistralAI(model_name="mistral-small-2506")
tools = {"get_weather": get_weather, "get_news": get_news}
llm_tool = llm.bind_tools([get_weather, get_news])

# -------------------------------
# Interactive Loop
# -------------------------------
def run_city_intelligence():
    print("[bold green]City Intelligence System[/]")
    print("Type 'exit' to quit\n")

    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        messages.append(HumanMessage(user_input))

        while True:
            result = llm_tool.invoke(messages)
            messages.append(result)

            if result.tool_calls:
                for tool_call in result.tool_calls:
                    tool_name = tool_call["name"]

                    confirm = input(f"Agent wants to call {tool_name}. Proceed? (y/n): ")
                    if confirm.lower() == "n":
                        print("[red]Tool call denied. Cannot fetch latest information.[/]")
                        break

                    # Execute tool
                    tool_result = tools[tool_name].invoke(tool_call)
                    messages.append(
                        ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                    )
                continue
            else:
                print(result.content)
                break


# -------------------------------
# Demo Run
# -------------------------------
if __name__ == "__main__":
    print(get_news.invoke("Hyderabad"))
    print(get_weather.invoke("Hyderabad"))
    run_city_intelligence()
