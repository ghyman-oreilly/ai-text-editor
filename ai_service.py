from dotenv import load_dotenv
import logging
from pydantic import BaseModel
import os
import random
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
    def __init__(
            self, 
            responses_model="gpt-4o",
            openai_embedding_model='text-embedding-3-small',
            st_embedding_model='BAAI/bge-small-en-v1.5'
            ):
        from sentence_transformers import SentenceTransformer
        self.responses_model = responses_model
        self.openai_embedding_model = openai_embedding_model
        self.st_embedding_model = SentenceTransformer(st_embedding_model)
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client

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

    def call_ai_service(self, prompt: Prompt, delay: float = 0.5, max_retries: int = 5):
        client = self._get_openai_client()

        for attempt in range(max_retries):
            try:
                response = client.responses.create(
                    model=self.responses_model,
                    input=prompt.as_messages()
                )
                time.sleep(delay)
                return response.output_text

            except Exception as e:
                wait = delay * (2 ** attempt)
                jittered_wait = wait * random.uniform(0.8, 1.2)
                logging.warning(f"Retrying after {jittered_wait:.1f}s due to error...")
                time.sleep(jittered_wait)

        logging.error("Max retries exceeded.")
        return None

    def generate_openai_embedding(
        self,
        input_text: str,
    ):
        """
        Generate text embedding
        """
        try:
            response = openai.embeddings.create(model=self.openai_embedding_model, input=input_text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error calling AI service or parsing response: {e}")
            return None

    def generate_st_embedding(
        self,
        input_text: str,
        normalize_embeddings: bool = True
    ):
        """
        Generate text embedding
        """
        try:
            embedding = self.st_embedding_model.encode(input_text, normalize_embeddings=normalize_embeddings)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None