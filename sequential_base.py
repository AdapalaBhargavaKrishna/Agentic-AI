import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph , START , END

load_dotenv()

class pipelinestate(TypedDict):
    raw_input : str
    edited_text : str
    script_text : str
    final_output : str

llm = ChatGroq(model = 'openai/gpt-oss-120b' , temperature = 0.7)

def editor_node(state : pipelinestate) -> dict:
    '''Stage 1 : Cleans up grammar , removes typos , and refines the tone.'''
    print('Stage 1')

    prompt = (
        "You are an expert, copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text. \n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {'edited_text' : response.content.strip()}

def scriptwriter_node(state: pipelinestate) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""
    print('Stage 2')

    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text and transform "
        "it into a highly engaging, punchy, conversational video script hook. Make it sound "
        "like a real person speaking passionately. Return only the script content. \n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}

def translator_node(state: pipelinestate) -> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""
    print('Stage 3')

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high "
        "Return only the final English text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}

graph = StateGraph(pipelinestate)

graph.add_node('editor' , editor_node)
graph.add_node('scriptwriter' , scriptwriter_node)
graph.add_node('translator' , translator_node)

graph.add_edge(START , 'editor')
graph.add_edge('editor' , 'scriptwriter')
graph.add_edge('scriptwriter' , 'translator')
graph.add_edge('scriptwriter' , END)

app = graph.compile()

result = app.invoke({
    'raw_input' : 'AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory.'
})

print(result['final_output'])