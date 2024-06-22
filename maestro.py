import streamlit as st
import os
from anthropic import Anthropic
import re
from datetime import datetime
import json
from tavily import TavilyClient
import time
import zipfile
import io

st.set_page_config(page_title="Maestro", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
    }
    .output-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Configuration")

# api key inputs
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")

@st.cache_resource
def get_anthropic_client(api_key):
    return Anthropic(api_key=api_key)

@st.cache_resource
def get_tavily_client(api_key):
    return TavilyClient(api_key=api_key)

# init clients
client = get_anthropic_client(anthropic_api_key) if anthropic_api_key else None
tavily_client = get_tavily_client(tavily_api_key) if tavily_api_key else None



MODELS = {
    "Claude 3 Opus": "claude-3-opus-20240229",
    "Claude 3 Sonnet": "claude-3-sonnet-20240229",
    "Claude 3 Haiku": "claude-3-haiku-20240307",
    "Claude 3.5 Sonnet": "claude-3-5-sonnet-20240620"
}

def calculate_subagent_cost(model, input_tokens, output_tokens):
    pricing = {
        "claude-3-opus-20240229": {"input_cost_per_mtok": 15.00, "output_cost_per_mtok": 75.00},
        "claude-3-haiku-20240307": {"input_cost_per_mtok": 0.25, "output_cost_per_mtok": 1.25},
        "claude-3-sonnet-20240229": {"input_cost_per_mtok": 3.00, "output_cost_per_mtok": 15.00},
        "claude-3-5-sonnet-20240620": {"input_cost_per_mtok": 3.00, "output_cost_per_mtok": 15.00},
    }
    input_cost = (input_tokens / 1_000_000) * pricing[model]["input_cost_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * pricing[model]["output_cost_per_mtok"]
    total_cost = input_cost + output_cost
    return total_cost

def opus_orchestrator(objective, file_content=None, previous_results=None, use_search=False):
    previous_results_text = "\n".join(previous_results) if previous_results else "None"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Based on the following objective{' and file content' if file_content else ''}, and the previous sub-task results (if any), please break down the objective into the next sub-task, and create a concise and detailed prompt for a subagent so it can execute that task. IMPORTANT!!! when dealing with code tasks make sure you check the code for errors and provide fixes and support as part of the next sub-task. If you find any bugs or have suggestions for better code, please include them in the next sub-task prompt. Please assess if the objective has been fully achieved. If the previous sub-task results comprehensively address all aspects of the objective, include the phrase 'The task is complete:' at the beginning of your response. If the objective is not yet fully achieved, break it down into the next sub-task and create a concise and detailed prompt for a subagent to execute that task.:\n\nObjective: {objective}" + ('\nFile content:\n' + file_content if file_content else '') + f"\n\nPrevious sub-task results:\n{previous_results_text}"}
            ]
        }
    ]
    if use_search:
        messages[0]["content"].append({"type": "text", "text": "Please also generate a JSON object containing a single 'search_query' key, which represents a question that, when asked online, would yield important information for solving the subtask. The question should be specific and targeted to elicit the most relevant and helpful resources. Format your JSON like this, with no additional text before or after:\n{\"search_query\": \"<question>\"}\n"})

    opus_response = client.messages.create(
        model=st.session_state.orchestrator_model,
        max_tokens=4096,
        messages=messages
    )

    response_text = opus_response.content[0].text
    cost = calculate_subagent_cost(st.session_state.orchestrator_model, opus_response.usage.input_tokens, opus_response.usage.output_tokens)
    st.session_state.total_cost += cost

    search_query = None
    if use_search:
        json_match = re.search(r'{.*}', response_text, re.DOTALL)
        if json_match:
            json_string = json_match.group()
            try:
                search_query = json.loads(json_string)["search_query"]
                response_text = response_text.replace(json_string, "").strip()
            except json.JSONDecodeError:
                st.warning("Error parsing JSON for search query. Skipping search query extraction.")

    return response_text, file_content, search_query, cost

def haiku_sub_agent(prompt, search_query=None, previous_haiku_tasks=None, use_search=False, continuation=False):
    if previous_haiku_tasks is None:
        previous_haiku_tasks = []

    continuation_prompt = "Continuing from the previous answer, please complete the response."
    system_message = "Previous Haiku tasks:\n" + "\n".join(f"Task: {task['task']}\nResult: {task['result']}" for task in previous_haiku_tasks)
    if continuation:
        prompt = continuation_prompt

    qna_response = None
    if search_query and use_search and tavily_client:
        try:
            qna_response = tavily_client.qna_search(query=search_query)
            st.info(f"Search Query: {search_query}")
            st.info(f"QnA response: {qna_response}")
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]

    if qna_response:
        messages[0]["content"].append({"type": "text", "text": f"\nSearch Results:\n{qna_response}"})

    try:
        haiku_response = client.messages.create(
            model=st.session_state.sub_agent_model,
            max_tokens=4096,
            messages=messages,
            system=system_message
        )

        response_text = haiku_response.content[0].text
        cost = calculate_subagent_cost(st.session_state.sub_agent_model, haiku_response.usage.input_tokens, haiku_response.usage.output_tokens)
        st.session_state.total_cost += cost

        if haiku_response.usage.output_tokens >= 4000:
            st.warning("Output may be truncated. Attempting to continue the response.")
            continuation_response_text, continuation_cost = haiku_sub_agent(prompt, search_query, previous_haiku_tasks, use_search, continuation=True)
            response_text += continuation_response_text
            cost += continuation_cost

        return response_text, cost
    except Exception as e:
        st.error(f"Error in haiku_sub_agent: {str(e)}")
        return None, 0

def opus_refine(objective, sub_task_results, filename, projectname, continuation=False):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Objective: " + objective + "\n\nSub-task results:\n" + "\n".join(sub_task_results) + "\n\nPlease review and refine the sub-task results into a cohesive final output. Add any missing information or details as needed. When working on code projects, ONLY AND ONLY IF THE PROJECT IS CLEARLY A CODING ONE please provide the following:\n1. Project Name: Create a concise and appropriate project name that fits the project based on what it's creating. The project name should be no more than 20 characters long.\n2. Folder Structure: Provide the folder structure as a valid JSON object, where each key represents a folder or file, and nested keys represent subfolders. Use null values for files. Ensure the JSON is properly formatted without any syntax errors. Please make sure all keys are enclosed in double quotes, and ensure objects are correctly encapsulated with braces, separating items with commas as necessary.\nWrap the JSON object in <folder_structure> tags.\n3. Code Files: For each code file, include ONLY the file name NEVER EVER USE THE FILE PATH OR ANY OTHER FORMATTING YOU ONLY USE THE FOLLOWING format 'Filename: <filename>' followed by the code block enclosed in triple backticks, with the language identifier after the opening backticks, like this:\n\n​python\n<code>\n​"}
            ]
        }
    ]

    opus_response = client.messages.create(
        model=st.session_state.refiner_model,
        max_tokens=4096,
        messages=messages
    )

    response_text = opus_response.content[0].text.strip()
    st.session_state.total_cost += calculate_subagent_cost(st.session_state.refiner_model, opus_response.usage.input_tokens, opus_response.usage.output_tokens)

    if opus_response.usage.output_tokens >= 4000 and not continuation:
        st.warning("Output may be truncated. Attempting to continue the response.")
        continuation_response_text = opus_refine(objective, sub_task_results + [response_text], filename, projectname, continuation=True)
        response_text += "\n" + continuation_response_text

    return response_text

def create_folder_structure(project_name, folder_structure, code_blocks):
    try:
        os.makedirs(project_name, exist_ok=True)
        st.success(f"Created project folder: {project_name}")
    except OSError as e:
        st.error(f"Error creating project folder: {project_name}\nError: {e}")
        return

    create_folders_and_files(project_name, folder_structure, code_blocks)

def create_folders_and_files(current_path, structure, code_blocks):
    for key, value in structure.items():
        path = os.path.join(current_path, key)
        if isinstance(value, dict):
            try:
                os.makedirs(path, exist_ok=True)
                st.success(f"Created folder: {path}")
                create_folders_and_files(path, value, code_blocks)
            except OSError as e:
                st.error(f"Error creating folder: {path}\nError: {e}")
        else:
            code_content = next((code for file, code in code_blocks if file == key), None)
            if code_content:
                try:
                    with open(path, 'w') as file:
                        file.write(code_content)
                    st.success(f"Created file: {path}")
                except IOError as e:
                    st.error(f"Error creating file: {path}\nError: {e}")
            else:
                st.warning(f"Code content not found for file: {key}")

def create_zip_file(project_name, folder_structure, code_blocks):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        def add_to_zip(current_path, structure):
            for key, value in structure.items():
                path = os.path.join(current_path, key)
                if isinstance(value, dict):
                    add_to_zip(path, value)
                else:
                    code_content = next((code for file, code in code_blocks if file == key), None)
                    if code_content:
                        zf.writestr(path, code_content)
    
        add_to_zip(project_name, folder_structure)
    
    memory_file.seek(0)
    return memory_file




st.title("🤖 Maestro 🤖")

st.sidebar.title("Model Selection")
st.session_state.orchestrator_model = st.sidebar.selectbox("Orchestrator Model", list(MODELS.values()), index=3)
st.session_state.sub_agent_model = st.sidebar.selectbox("Sub-agent Model", list(MODELS.values()), index=3)
st.session_state.refiner_model = st.sidebar.selectbox("Refiner Model", list(MODELS.values()), index=3)

objective = st.text_area("Enter your objective:", height=100)
use_file = st.checkbox("Add a text file")
file_content = None

if use_file:
    uploaded_file = st.file_uploader("Choose a file", type="txt")
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue().decode("utf-8")
        st.text_area("File content:", value=file_content, height=150)

use_search = st.checkbox("Use search")

if st.button("Start Task"):
    if not anthropic_api_key:
        st.error("Please provide a valid Anthropic API key to start the task.")
    elif not objective:
        st.error("Please enter an objective.")
    else:
        st.session_state.task_exchanges = []
        st.session_state.haiku_tasks = []
        st.session_state.total_cost = 0

        progress_bar = st.progress(0)
        status_text = st.empty()
        cost_text = st.empty()
        task_output = st.empty()

        with st.expander("Task Progress", expanded=True):
            while True:
                previous_results = [result for _, result in st.session_state.task_exchanges]
                if not st.session_state.task_exchanges:
                    status_text.text("Orchestrator is analyzing the objective...")
                    opus_result, file_content_for_haiku, search_query, opus_cost = opus_orchestrator(objective, file_content, previous_results, use_search)
                else:
                    status_text.text("Orchestrator is planning the next step...")
                    opus_result, _, search_query, opus_cost = opus_orchestrator(objective, previous_results=previous_results, use_search=use_search)

                if opus_result is None:
                    st.error("Orchestrator encountered an error. Stopping the task.")
                    break

                cost_text.text(f"Current total cost: ${st.session_state.total_cost:.4f}")
                task_output.markdown(f"**Orchestrator output:**\n{opus_result}")
                time.sleep(1)  # Simulate processing time

                if "The task is complete:" in opus_result:
                    final_output = opus_result.replace("The task is complete:", "").strip()
                    break
                else:
                    sub_task_prompt = opus_result
                    if file_content_for_haiku and not st.session_state.haiku_tasks:
                        sub_task_prompt = f"{sub_task_prompt}\n\nFile content:\n{file_content_for_haiku}"

                    status_text.text("Sub-agent is working on the task...")
                    sub_task_result, haiku_cost = haiku_sub_agent(sub_task_prompt, search_query, st.session_state.haiku_tasks, use_search)
                    
                    if sub_task_result is None:
                        st.error("Sub-agent encountered an error. Stopping the task.")
                        break

                    st.session_state.haiku_tasks.append({"task": sub_task_prompt, "result": sub_task_result})
                    st.session_state.task_exchanges.append((sub_task_prompt, sub_task_result))
                    file_content_for_haiku = None

                    task_output.markdown(f"**Sub-agent result:**\n{sub_task_result}")
                    cost_text.text(f"Current total cost: ${st.session_state.total_cost:.4f}")

                progress = len(st.session_state.task_exchanges) / (len(st.session_state.task_exchanges) + 1)
                progress_bar.progress(progress)

        status_text.text("Refining the final output...")
        task_output.empty()
        time.sleep(1)

        sanitized_objective = re.sub(r'\W+', '_', objective)
        timestamp = datetime.now().strftime("%H-%M-%S")
        refined_output = opus_refine(objective, [result for _, result in st.session_state.task_exchanges], timestamp, sanitized_objective)

        progress_bar.progress(1.0)
        status_text.text("Task completed!")
        cost_text.text(f"Final total cost: ${st.session_state.total_cost:.4f}")

        st.subheader("Refined Output")
        st.markdown(refined_output)

        project_name_match = re.search(r'Project Name: (.*)', refined_output)
        project_name = project_name_match.group(1).strip() if project_name_match else sanitized_objective

        folder_structure_match = re.search(r'<folder_structure>(.*?)</folder_structure>', refined_output, re.DOTALL)
        folder_structure = {}
        if folder_structure_match:
            json_string = folder_structure_match.group(1).strip()
            try:
                folder_structure = json.loads(json_string)
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON: {e}")
                st.error(f"Invalid JSON string: {json_string}")

        code_blocks = re.findall(r'Filename: (\S+)\s*```[\w]*\n(.*?)\n```', refined_output, re.DOTALL)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Project Files"):
                create_folder_structure(project_name, folder_structure, code_blocks)

        with col2:
            if folder_structure and code_blocks:
                zip_file = create_zip_file(project_name, folder_structure, code_blocks)
                st.download_button(
                    label="Download Project as ZIP",
                    data=zip_file,
                    file_name=f"{project_name}.zip",
                    mime="application/zip"
                )

        st.subheader("Task Breakdown")
        for i, (prompt, result) in enumerate(st.session_state.task_exchanges, start=1):
            with st.expander(f"Task {i}"):
                st.write("**Prompt:**")
                st.write(prompt)
                st.write("**Result:**")
                st.write(result)

        exchange_log = f"Objective: {objective}\n\n"
        exchange_log += "=" * 40 + " Task Breakdown " + "=" * 40 + "\n\n"
        for i, (prompt, result) in enumerate(st.session_state.task_exchanges, start=1):
            exchange_log += f"Task {i}:\n"
            exchange_log += f"Prompt: {prompt}\n"
            exchange_log += f"Result: {result}\n\n"

        exchange_log += "=" * 40 + " Refined Final Output " + "=" * 40 + "\n\n"
        exchange_log += refined_output

        max_length = 25
        truncated_objective = sanitized_objective[:max_length] if len(sanitized_objective) > max_length else sanitized_objective

        filename = f"{timestamp}_{truncated_objective}.md"

        st.download_button(
            label="Download Full Exchange Log",
            data=exchange_log,
            file_name=filename,
            mime="text/markdown"
        )

if __name__ == "__main__":
    st.sidebar.info("This app uses AI models to break down and complete complex tasks. Enter your objective, optionally add a file or use search, and watch as the AI orchestrates the solution! credits to https://github.com/Doriandarko/maestro")
