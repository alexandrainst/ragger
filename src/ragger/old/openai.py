"""Functions and classes related to calling the OpenAI model API."""

# import os
# import sys
#
# import openai
# from azure.core.credentials import AccessToken
# from langchain.chat_models import AzureChatOpenAI
#
#
# def get_azure_chat_openai(
#     temperature: float = 0.0, streaming: bool = False
# ) -> AzureChatOpenAI:
#     """Initialise the OpenAI Chat Model, running on Azure.
#
#     This requires the `OPENAI_API_HOST` and `OPENAI_API_KEY` environment
#     variables to be set.
#
#     Args:
#         temperature:
#             The amount of creativity allowed when generating text. A temperature of 0.0
#             will be close to deterministic.
#         streaming:
#             Whether to use the streaming API or not.
#
#     Returns:
#         The OpenAI Chat Model.
#     """
#     # Set Open AI class variables needed to access the Azure LLM
#     openai.api_type = os.environ["AZURE_DEPLOYMENT_ID"]
#     openai.api_version = os.environ["OPENAI_API_VERSION"]
#     openai_api_base = os.environ["OPENAI_API_HOST"]
#     openai_api_key = AccessToken(
#         token=os.environ["OPENAI_API_KEY"], expires_on=sys.maxsize
#     ).token
#
#     llm = AzureChatOpenAI(
#         temperature=temperature,
#         model=os.environ["DEFAULT_MODEL"],
#         openai_api_version=openai.api_version,
#         openai_api_base=openai_api_base,
#         openai_api_key=openai_api_key,
#         deployment_name=os.environ["AZURE_DEPLOYMENT_ID"],
#         streaming=streaming,
#     )
#     return llm
