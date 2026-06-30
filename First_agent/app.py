from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import json
import re
import requests
import pytz
import yaml
from pathlib import Path
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def my_custom_tool(arg1:str, arg2:int)-> str: #it's import to specify the return type
    #Keep this format for the description / args / args description but feel free to modify the tool
    """A tool that does nothing yet 
    Args:
        arg1: the first argument
        arg2: the second argument
    """
    return "What magic will you build ?"

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)
web_search_tool = DuckDuckGoSearchTool()


@tool
def inspect_image_generation_tool() -> str:
    """Inspect the image generation tool's metadata so you can model new tools after it."""
    tool_summary = {
        "name": getattr(image_generation_tool, "name", None),
        "description": getattr(image_generation_tool, "description", None),
        "inputs": getattr(image_generation_tool, "inputs", None),
        "output_type": getattr(image_generation_tool, "output_type", None),
    }
    return json.dumps(tool_summary, indent=2)


@tool
def visit_webpage(url: str) -> str:
    """Visit a webpage and return its readable text as markdown.
    Args:
        url: The webpage URL to read.
    """
    try:
        from markdownify import markdownify
        from smolagents.utils import truncate_content
    except ImportError as e:
        return f"Missing dependency for visit_webpage: {e}"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return truncate_content(markdown_content, 10000)
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again later or check the URL."
    except requests.exceptions.RequestException as e:
        return f"Error fetching the webpage: {str(e)}"


@tool
def wiki(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary.
    Args:
        query: The Wikipedia topic or search query.
    """
    try:
        search_response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 1,
            },
            timeout=20,
        )
        search_response.raise_for_status()
        results = search_response.json().get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia result found for: {query}"

        title = results[0]["title"]
        summary_response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=20,
        )
        summary_response.raise_for_status()
        summary = summary_response.json()
        extract = summary.get("extract", "No summary available.")
        page_url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
        return f"{title}: {extract}\n{page_url}".strip()
    except requests.exceptions.RequestException as e:
        return f"Error fetching Wikipedia result: {str(e)}"


@tool
def translator(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text from one language to another.
    Args:
        text: The text to translate.
        src_lang: The source language, for example 'French'.
        tgt_lang: The target language, for example 'English'.
    """
    if src_lang.lower() == tgt_lang.lower():
        return text

    try:
        response = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text, "langpair": f"{src_lang}|{tgt_lang}"},
            timeout=20,
        )
        response.raise_for_status()
        translated_text = response.json().get("responseData", {}).get("translatedText")
        if translated_text:
            return translated_text
    except requests.exceptions.RequestException:
        pass

    return (
        "Translation service is unavailable or did not understand the language pair. "
        f"Original text: {text}"
    )


@tool
def document_qa(document: str, question: str) -> str:
    """Answer a question about a text document using simple keyword matching.
    Args:
        document: The document text, or a path to a local text/markdown file.
        question: The question to answer from the document.
    """
    try:
        document_path = Path(document)
        is_text_file = len(document) < 300 and document_path.exists() and document_path.is_file()
    except OSError:
        is_text_file = False

    if is_text_file:
        document_text = document_path.read_text(encoding="utf-8", errors="ignore")
    else:
        document_text = document

    question_terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z0-9']+", question)
        if len(term) > 3
    }
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", document_text) if paragraph.strip()]
    ranked_paragraphs = sorted(
        paragraphs,
        key=lambda paragraph: sum(term in paragraph.lower() for term in question_terms),
        reverse=True,
    )

    if ranked_paragraphs and any(term in ranked_paragraphs[0].lower() for term in question_terms):
        return ranked_paragraphs[0]
    return "I could not find a clear answer in the document."


@tool
def image_qa(image: str, question: str) -> str:
    """Answer a question about an image.
    Args:
        image: A local image path or image URL.
        question: The question to answer about the image.
    """
    return (
        "image_qa is registered, but no vision model is configured in this app yet. "
        f"Image: {image}. Question: {question}"
    )


with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        my_custom_tool,
        get_current_time_in_timezone,
        image_generation_tool,
        inspect_image_generation_tool,
        web_search_tool,
        visit_webpage,
        wiki,
        translator,
        document_qa,
        image_qa,
    ], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()
