from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

prompt = PromptTemplate.from_template(
 "You are a senior backend engineer. Explain {topic} clearly."
)


chain=prompt|llm


while True:
    user_input=input("Enter your qustions")
    if user_input.lower()=="exit":
        break
    else:
        response = chain.invoke(user_input)

        if isinstance(response.content, list):
            text = "".join(
                block["text"]
                
                for block in response.content
                if block.get("type") == "text"
            )
        else:
            text = response.content
        print(text)
            

