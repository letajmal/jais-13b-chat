data = {
  "inputs": {
    "chat_history": ["Hi! How are you doing today?", "I'm great! What would you like to learn about?"],
    "text": "Can you tell me some interesting facts about France?"
  }
}

response = handler(data)

print(response["response"])