from dotenv import load_dotenv
import logging
from pydantic import BaseModel
import openai
import os
import time
from typing import Optional, List, Dict, Union, Literal


# load env variables from .env
load_dotenv()

# init logger
logger = logging.getLogger(__name__)

# initialize client
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "The environment variable OPENAI_API_KEY is not set. "
        "Please set it to your OpenAI API key before running this script."
    )

client = openai.OpenAI(api_key=api_key)

class PromptContent(BaseModel):
    type: Union[Literal['input_text'], Literal['input_image']]
    text: Optional[str] = None
    image_url: Optional[str] = None
    detail: Optional[str] = None

class SystemRole(BaseModel):
    role: str = "system"
    content: str

class UserRole(BaseModel):
    role: str = "user"
    content: list[PromptContent]
    
class Prompt(BaseModel):
    user_role: UserRole
    system_role: Optional[SystemRole] = None

    def as_messages(self) -> List[Dict[str, str]]:
        messages = []
        if self.system_role:
            messages.append(self.system_role.model_dump(exclude_none=True))
        messages.append(self.user_role.model_dump(exclude_none=True))
        return messages


class AIServiceCaller:
    def __init__(self, model="gpt-4o"):
        self.model = model

    def create_prompt_object(
            self, 
            user_role_text_content: str,
            user_role_img_content: Optional[str] = None, # data uri
            system_role_content: str = None,
            detail: str = 'high'
        ):
        """
        Build the prompt for an AI service call.
        """       
        if not user_role_text_content:
            return None
        
        prompt_user_content = []

        prompt_user_content.append(PromptContent(
            type='input_text',
            text=user_role_text_content
        ))

        if user_role_img_content:
            prompt_user_content.append(PromptContent(
                type='input_image',
                image_url=user_role_img_content,
                detail=detail
            ))
        
        if system_role_content:
            return Prompt(
                user_role=UserRole(content=prompt_user_content),
                system_role=SystemRole(content=system_role_content)
            )

        return Prompt(
                user_role=UserRole(content=prompt_user_content)
            )

    def call_ai_service(
            self, 
            prompt: Prompt,
            delay: int = 0.5,
    ):
        """
        Call OpenAI responses API.
        """
        try:
            response = client.responses.create(
                model=self.model,
                input=prompt.as_messages()
            )
            
            time.sleep(delay)

            return response.output_text
        except Exception as e:
            logger.error(f"Error calling AI service: {e}")
            return None
