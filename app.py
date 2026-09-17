from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)


while True:
    user_input=input("Enter your qustions")
    if user_input.lower()=="exit":
        break
    else:
        response = llm.invoke(user_input)

        if isinstance(response.content, list):
            text = "".join(
                block["text"]
                
                for block in response.content
                if block.get("type") == "text"
            )
        else:
            text = response.content

        print(text)
            

