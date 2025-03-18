
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator

def general_agent(chat_input, chat_history, llmconfig):
    
    class general_response(BaseModel):
        response: str = Field(description = "general assinatant response")
        other_agent: bool = Field(description = "If assistance from other agent is needed")

    system_prompt = f"""
        You are BostonTech AI Assistant, a helpful and knowledgeable AI designed to assist customers with their PC hardware needs.\
        Your primary role is to provide information about the store and assist users with general inquiries.

        store_name : BostonTech
        store_location : 123 Tech Street, Boston, MA
        store_hours : Monday-Saturday: 9 AM - 8 PM, Sunday: 10 AM - 6 PM

        **Tone & Response Style:**

        - Maintain a professional yet friendly, engaging tone, concise, to the point.

        **Handling Other Queries:**

        - If the user asks about billing, general FAQs, or requests to speak with a human representative, gracefully redirect them to the appropriate department.
        - Example: "For billing and general inquiries, I’ll connect you with the right representative. Please hold on."

        **Conversational Flow:**

        - If a user asks about the store, provide relevant details while keeping responses natural and engaging.
        - If the user is unsure of what they need, ask guiding questions to understand their intent better.
        - If a query is outside the AI's scope, politely inform the user and, if possible, guide them to an alternative resource.
        
        chat history for context: {chat_history}
    """

    parser = PydanticOutputParser(pydantic_object= general_response)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    return
