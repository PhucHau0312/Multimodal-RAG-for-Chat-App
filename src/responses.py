import logging
import os
import traceback

from openai import OpenAI
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Literal

from src.prompts import routing_prompt, visual_prompt, textual_prompt, combined_prompt 


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default=None)
LLM_ENDPOINT = os.environ.get("VLM_ENDPOINT", "")  ## 
logger = logging.getLogger(__name__)


client = OpenAI(api_key=OPENAI_API_KEY)
vlm = OpenAI(base_url=LLM_ENDPOINT, 
            api_key="EMPTY" 
            )

class QueryProcessor(BaseModel):
    route: Literal["chitchat", "rag"] = Field(
        description="Choose 'rag' if the query requires looking up external documents or specific data. Choose 'chitchat' for greetings, jokes, or general small talk."
    )
    rewritten_query: str = Field(
        description="A standalone version of the user's query that includes all necessary context from the chat history. If routing is chitchat, this can be a cleaned-up version of the input."
    )
    
def openai_chat_complete(messages, response_format=None, model="gpt-4o-mini", raw=False):
    if response_format: 
        response = client.beta.chat.completions.parse(
            model=model, 
            messages=messages,
            response_format=response_format,
        )

        return response.choices[0].message.parsed
    
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        if raw:
            return response.choices[0].message
        output = response.choices[0].message

        return output.content


def vlm_chat_complete(messages, model="custom_model", raw=False):
    response = vlm.chat.completions.create(
        model=model,
        messages=messages
    )
    if raw:
        return response.choices[0].message
    output = response.choices[0].message
    return output.content


def route_and_rewrite(query, chat_history):
    formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])

    messages = [
                {"role": "system", "content": routing_prompt},
                {"role": "user", "content": f"CHAT_HISTORY:\n{formatted_history}\n\nUSER_QUERY: {query}"}
                ]

    return openai_chat_complete(messages=messages, response_format=QueryProcessor)
    
    
async def generate_visual_response(query, visual_contexts):
    try:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{page.get("image").get("base64")}"
                }
            } for page in visual_contexts

        ] + [{"type": "text", "text": visual_prompt.format(query)}]

        messages = [{"role": "user", "content": content}]

        return vlm_chat_complete(messages=messages)
    
    except Exception as e:
            logger.error(f"Error generating visual response: {str(e)}")
            traceback.print_exc()
            return "Error generating response from visual contexts."


async def generate_textual_response(query, textual_contexts): 
    try:
        contexts = [ctx["text"]["content"] for ctx in textual_contexts]
        contexts_str = "\n- ".join(contexts)

        messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": textual_prompt.format(query, contexts_str)}]
            }
            ]

        return vlm_chat_complete(messages=messages)
    
    except Exception as e:
            logger.error(f"Error generating textual response: {str(e)}")
            return "Error generating response from textual contexts."
    

async def combine_responses(query, chat_history, visual_response, textual_response):
    """
    Combine visual and textual responses to generate a final answer.
    
    Args:
        query (str): The user's question
        chat_history (str): The chat history
        visual_response (dict): Response from visual contexts
        textual_response (dict): Response from textual contexts
        answer (str): Ground truth answer if available
        
    Returns:
        dict: Combined response
    """
    try:
        visual_evidence = visual_response.get('Evidence', "Evidence not available")
        visual_thought = visual_response.get('Chain of Thought', "CoT not available")
        visual_answer = visual_response['Answer']
        textual_evidence = textual_response.get('Evidence', "Evidence not available")
        textual_thought = textual_response.get('Chain of Thought', "CoT not available")
        textual_answer = textual_response['Answer']
        
        prompt = combined_prompt.format(
            query=query,
            chat_history=chat_history,
            visual_evidence=visual_evidence,
            visual_thought=visual_thought,
            visual_answer=visual_answer,
            textual_evidence=textual_evidence,
            textual_thought=textual_thought,
            textual_answer=textual_answer,
        )

        messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }
            ]

        return vlm_chat_complete(messages=messages)
    
    except Exception as e:
        logger.error(f"Error combining responses: {str(e)}")
        return {
            "Analysis": "Error occurred during analysis.",
            "Conclusion": "Error occurred during conclusion.",
            "Final Answer": "Error occurred during combination of responses."
        }
    

async def combine_responses_for_single_query(query, visual_response, textual_response):

    instruction_prompt = """
    Analyze the following two responses to the question: "{query}"

    Response 1:
    Evidence: {visual_evidence}
    Chain of Thought: {visual_thought}
    Final Answer: {visual_answer}

    Response 2:
    Evidence: {textual_evidence}
    Chain of Thought: {textual_thought}
    Final Answer: {textual_answer}

    Response 1 is based on a visual q/a pipeline, and Response 2 is based on a textual q/a pipeline. 
    - In general, given both response 1 and response 2 have logical chains of thoughts, and decision boils down to evidence, you should place higher degree of trust on evidence reported in Response 1.
    - If one of the responses has declined giving a clear answer, please weigh the other answer more unless there is reasonable thought to not answer, and both thoughts are inconsistent.
    - Language of the answer should be short and direct, usually answerable in a single sentence, or phrase. You should directly give the specific response to an answer.

    Consider both chains of thought and final answers. Provide your analysis in the following format:

    ## Analysis:
    [Your detailed analysis here, evaluating the consistency of both the chains of thoughts, with respect to each other, the question and their respective answers, as well as validity of the evidence.]

    ## Conclusion:
    [Your conclusion on which answer is more likely to be correct, or if a synthesis of both is needed]

    ## Final Answer:
    [Answer the question "{query}", based on your analysis of the two candidates so far. Please ensure that answers are short and concise, similar in language to the provided answers.]
    """

    try:
        visual_evidence = visual_response.get('Evidence', "Evidence not available")
        visual_thought = visual_response.get('Chain of Thought', "CoT not available")
        visual_answer = visual_response['Answer']
        textual_evidence = textual_response.get('Evidence', "Evidence not available")
        textual_thought = textual_response.get('Chain of Thought', "CoT not available")
        textual_answer = textual_response['Answer']
        
        prompt = instruction_prompt.format(
            query=query,
            visual_evidence=visual_evidence,
            visual_thought=visual_thought,
            visual_answer=visual_answer,
            textual_evidence=textual_evidence,
            textual_thought=textual_thought,
            textual_answer=textual_answer,
        )

        messages = [
            {
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            }
            ]

        return vlm_chat_complete(messages=messages)
    
    except Exception as e:
        logger.error(f"Error combining responses: {str(e)}")
        return {
            "Analysis": "Error occurred during analysis.",
            "Conclusion": "Error occurred during conclusion.",
            "Final Answer": "Error occurred during combination of responses."
        }