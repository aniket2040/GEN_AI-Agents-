from rich import print
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
import os
import requests
from langchain.agents import create_agent, middleware
from langchain.agents.middleware import wrap_tool_call

# -------------------------------
# Weather Tool
# -------------------------------
# =========================
# 🌦️ Weather Tool
# =========================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


# =========================
# 📰 News Tool (Tavily)
# =========================

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""

    response = tavily_client.search(
        query=f"latest news in {city}",
        search_depth="basic",
        max_results=3
    )

    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []

    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")

        news_list.append(
            f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


llm = ChatMistralAI(model_name="mistral-small-2506")
@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agent wants to call '{tool_name}'. Approve? (yes/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id=request.tool_call["id"]
        )

    return handler(request)

agent = create_agent(
    llm,
    tools = [get_weather,get_news],
    system_prompt= "you are helpful city assistant",
    middleware = [human_approval]

)

print("City Agent  |  type exit to exit")
while True:
    user = input("you  : ")
    if user.lower() == "exit":
        break

    result = agent.invoke({
        "message": [
            {
                "role":"user",
                "content": user
            }
        ]
    })
    print("bot : ",result["message"][-1].content)