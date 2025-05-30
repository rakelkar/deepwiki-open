from api.openai_client import OpenAIClient
import os
from openai import AzureOpenAI, AsyncAzureOpenAI
import httpx

class AzureOpenAIClient(OpenAIClient):
    def __init__(self, **kwargs):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", None)
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", None)
        super().__init__(**kwargs)

    def init_sync_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        http_client = httpx.Client(timeout=5)
        return AzureOpenAI(azure_endpoint = self.azure_endpoint, 
            api_key=api_key,  
            api_version=self.azure_api_version,
            http_client=http_client
            )

    def init_async_client(self):
        api_key = self._api_key or os.getenv(self._env_api_key_name)
        if not api_key:
            raise ValueError(
                f"Environment variable {self._env_api_key_name} must be set"
            )
        http_async_client = httpx.AsyncClient(timeout=5)
        return AsyncAzureOpenAI(azure_endpoint = self.azure_endpoint, 
            api_key=api_key,  
            api_version=self.azure_api_version,
            http_client=http_async_client
            )
