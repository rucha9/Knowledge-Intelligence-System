from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationRetrievalChain
from langchain.memory import ConversationBufferMemory
from app.config import Config

class LLMService:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=Config.OPENAI_API_KEY
        )

    
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )

        self.chain = ConversationRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory
        )

    def get_response(self, query):
        try:
            response = self.chain({"question": query})
            return response['answer']
        except Exception as e:
            print(f"Error occurred: {e}")
            return "Sorry, I encountered an error while processing your request."