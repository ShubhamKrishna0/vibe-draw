import openai
from app.core.config import settings
from app.core.celery_app import celery_app
from app.tasks.tasks import AsyncAITask, GenericPromptTask, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

openai.api_key = settings.OPENAI_API_KEY

class OpenAIPromptTask(GenericPromptTask, AsyncAITask):
    async def send_message(self, prompt: str, system_prompt=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )

        return {
            "content": response['choices'][0]['message']['content'],
            "usage": response['usage']
        }

OpenAIPromptTask = celery_app.register_task(OpenAIPromptTask())
