# basic_dubai_faq.py

from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
import fuzzysearch

# --- Step 1: Make a simple FAQ dataset (list of dicts) ---
faqs = [
    {"question": "What is the currency in Dubai?", "answer": "The currency is the UAE Dirham (AED)."},
    {"question": "What is the best time to visit Dubai?", "answer": "The best time is from November to March when the weather is cooler."},
    {"question": "Do I need a visa to visit Dubai?", "answer": "Visa requirements depend on your nationality. Many countries get visa on arrival."},
    {"question": "What language is spoken in Dubai?", "answer": "Arabic is the official language, but English is widely spoken."},
    {"question": "What is the tallest building in Dubai?", "answer": "The Burj Khalifa, at 828 meters, is the tallest building."},
    {"question": "Is Dubai safe for tourists?", "answer": "Yes, Dubai is considered one of the safest cities in the world."},
    {"question": "Can I drink alcohol in Dubai?", "answer": "Yes, but only in licensed venues like hotels and clubs."},
    {"question": "What is the dress code in Dubai?", "answer": "Modest dress is recommended in public, but swimwear is fine at beaches/pools."},
    {"question": "Is public transport available in Dubai?", "answer": "Yes, there is metro, buses, and taxis."},
    {"question": "What is the Dubai Metro?", "answer": "A modern, driverless metro system covering many key areas of the city."},
    {"question": "What are famous shopping places in Dubai?", "answer": "The Dubai Mall, Mall of the Emirates, and traditional souks."},
    {"question": "What is Dubai Expo City?", "answer": "It is the legacy site of Expo 2020, now an innovation hub."},
    {"question": "Can I use credit cards in Dubai?", "answer": "Yes, credit cards are widely accepted."},
    {"question": "What is the weekend in Dubai?", "answer": "The weekend is Saturday and Sunday."},
    {"question": "What power plugs are used in Dubai?", "answer": "Type G plugs with 230V supply."},
    {"question": "Is Dubai expensive?", "answer": "It can be expensive, but there are also budget-friendly options."},
    {"question": "What is the Dubai Frame?", "answer": "A landmark structure shaped like a giant picture frame."},
    {"question": "Is Uber available in Dubai?", "answer": "Yes, Uber and Careem operate in Dubai."},
    {"question": "What is the time zone of Dubai?", "answer": "Dubai is in Gulf Standard Time (GMT+4)."},
    {"question": "What is the Dubai Fountain?", "answer": "A choreographed fountain show outside the Dubai Mall and Burj Khalifa."}
]

# --- Step 2: Define a helper function for fuzzy matching ---
def search_faq(user_question):
    best_match = None
    best_score = -1
    for item in faqs:
        # Fuzzysearch returns matches (position, etc.), we just check if it finds substring similarity
        matches = fuzzysearch.find_near_matches(user_question.lower(), item["question"].lower(), max_l_dist=3)
        if matches:  # found a close match
            score = len(matches[0].matched)
            if score > best_score:
                best_score = score
                best_match = item
    return best_match

# --- Step 3: Define LangGraph Node ---
def faq_agent(state: MessagesState):
    user_msg = state["messages"][-1].content
    match = search_faq(user_msg)
    if match:
        return {"messages": [AIMessage(content=match["answer"])]}
    else:
        return {"messages": [AIMessage(content="Sorry, I don’t know the answer.")]}

# --- Step 4: Build the LangGraph workflow ---
graph = StateGraph(MessagesState)
graph.add_node("faq_agent", faq_agent)
graph.set_entry_point("faq_agent")
faq_workflow = graph.compile()

# --- Step 5: Run in terminal loop ---
if __name__ == "__main__":
    print("🤖 Dubai FAQ Bot (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break
        result = faq_workflow.invoke({"messages": [HumanMessage(content=user_input)]})
        print("Bot:", result["messages"][-1].content)
