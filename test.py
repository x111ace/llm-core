# To run this test:
# 1. Ensure you are in the root `llm-core` directory.
# 2. Build and install the Rust library into a conda environment:
#    python llm_core/__rsdl__.py -i
# 3. Activate the new conda environment and run this test script:
#    conda activate llm_core_venv
#    python test.py

import os, sys, json

from llm_core import (
    Chat,
    Tool,
    ToolDefinition,
    SimpleSchema,
    SchemaProperty,
    run_sorter,
    SortingInstructions,
)

MODEL_NAME = "GEMINI 2.0 FLASH"



def chat_comm():
    exit_comms =  [
        'exit', 'e',
        'quit', 'q',
        'stop', 's',
        'x', 'o', '0'
    ]
    help_comms = [
        'help', 'h',
    ]
    return exit_comms, help_comms

def help_menu():
    (
        exit_comms,
        help_comms
    ) = chat_comm(); print("\n" +
        "--- HELP MENU ---" + "\n\n" +

        "Type " + str(help_comms) + " to show this menu again." + "\n\n" +

        "Exit Commands: " + str(exit_comms) + "\n")

    return exit_comms, help_comms



def init_data(
        input_file, 
        output_dir
    ) -> tuple[str, str]:
    if not os.path.exists(input_file):
        print(f"\n[ERROR] Test data file not found at: {input_file}", file=sys.stderr)
        print("Please create this file with a JSON array of strings to sort.", file=sys.stderr)
        return None, None
    else:
        return input_file, output_dir

def sort_data(
        data_items: list[str] = None,
        input_file: str = None, 
        output_dir: str = None,
        model_name: str = MODEL_NAME,
        instructions: SortingInstructions = None,
        swarm_size: int = 1,
        debug_out: bool = False
    ):
    if (data_items is None and input_file is None) or \
       (data_items is not None and input_file is not None):
        print(f"\n[ERROR] Provide either data_items or an input_file, but not both.", file=sys.stderr)
        return
    
    current_instructions = instructions if instructions is not None else SORTING_INSTRUCTIONS
    
    try:
        if data_items is not None:
            # 3. Run the sorter with data items.
            sorted_results = run_sorter(
                model_name=model_name,
                items_list=data_items,   
                instructions=current_instructions,
                output_path=output_dir,
                swarm_size=swarm_size,
                debug_out=debug_out
            )
        else:
            # Run the sorter with input file.
            sorted_results = run_sorter(
                model_name=model_name,
                input_path=input_file,
                instructions=current_instructions,
                output_path=output_dir,
                swarm_size=swarm_size,
                debug_out=debug_out
            )
        
        # Print the results.
        print("\n--- SORTER RESULTS ---\n")
        print(json.dumps(sorted_results, indent=2))
        print(f"\nFull results also saved to a new file in the '{output_dir}' directory.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during sorting: {e}", file=sys.stderr)



def init_chat(
        model_name: str = None,
        system_prompt: str = None,
        native_tools: bool = False,
        extra_tools: list = None,
        schema: SimpleSchema = None,
        debug_out: bool = False
    ):
    try:
        if model_name is None:
            model_name = MODEL_NAME
        # 1. Initialize the Chat session from Rust.
        chat_session = Chat(
            model_name=model_name,
            system_prompt=system_prompt,
            native_tools=native_tools,
            extra_tools=extra_tools,
            schema=schema,
            debug_out=debug_out
        )
        return chat_session

    except Exception as e:
        print(f"\n[ERROR] Failed to initialize chat session: {e}", file=sys.stderr)
        print("Please ensure the model name is correct and your API keys are set in .env")
        return None

def chat_wrap(
        model_name: str = None,
        system_prompt: str = None,
        native_tools: bool = False,
        extra_tools: list = None,
        schema: SimpleSchema = None,
        debug_out: bool = False
    ):
    chat_session = init_chat(
        model_name=model_name,
        system_prompt=system_prompt,
        native_tools=native_tools,
        extra_tools=extra_tools,
        schema=schema,
        debug_out=debug_out
    )
    if not chat_session:
        return
    print("Chat session Initialized. Type 'exit' to end.")
    chat_loop(chat_session)

def chat_loop(chat_session: Chat):
    (exit_comms, help_comms
    ) = help_menu(); print("---\n")
    while True:
        try:
            user_prompt = input("INPUT: ")
            if user_prompt.lower() in exit_comms:
                break
            elif user_prompt.lower() in help_comms:
                help_menu()
                continue

            assistant_message = chat_session.send(user_prompt)
            if assistant_message and assistant_message.content:
                print(f"AGENT: {assistant_message.content.strip()}")

        except Exception as e:
            print(f"\n[ERROR] An error occurred during the chat: {e}", file=sys.stderr)
            break
    print("\n--- CHAT OVER ---\n")









SYSTEM_PROMPT = "You are a helpful and concise assistant."

def basics_chat():
    """
    Runs an interactive chat session using the Rust-powered Chat session manager.
    """
    print(
        "--- BASICS CHAT ---" + "\n\n" +
        "Provide a sentence with a name and age for the agent to extract." + "\n"
    ); chat_wrap(
        model_name=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
    )









USER_DETAILS_SCHEMA_SYSTEM_PROMPT = """You are a data entry assistant. 
            
Your only job is to extract user details into a JSON format based on the provided schema. 

Do not talk to the user."""

USER_DETAILS_SCHEMA = SimpleSchema(
    name="user_details",
    description="Extracts user details from a sentence.",
    properties=[
        SchemaProperty(name="name", 
            property_type="string", description="The user's first name."
        ),
        SchemaProperty(name="age", 
            property_type="integer", description="The user's age."
        ),
    ]
)

def schema_chat():
    """
    Runs an interactive chat session using the Rust-powered Chat session manager.
    """
    print(
        "--- SCHEMA CHAT ---" + "\n\n" +
        "Provide a sentence with a name and age for the agent to extract." + "\n"
    ); chat_wrap(
        model_name=MODEL_NAME,
        system_prompt=USER_DETAILS_SCHEMA_SYSTEM_PROMPT,
        schema=USER_DETAILS_SCHEMA
    )







### ### ###

def get_user_info() -> dict:
    """
    A simple Python tool that simulates fetching a hardcoded user profile
    from an external source, like a database. It takes no arguments.
    """
    # These values are hardcoded to simulate fetching data.
    name = "John"; age = "30"; return {
        "user_name": name,
        "user_age": age,
        "source": "python",
        "notes": "This data is hardcoded for simulation."
    }

### ### ###

def tool_defs():
    """
    Defines the tools that the agent can use.
    """
    get_user_info_tool = Tool(
        definition=ToolDefinition(
            name="get_user_info",
            description="Gets the current logged-in user's profile information.",
            parameters=SimpleSchema(
                name="get_user_info",
                description="Gets the current logged-in user's profile information.",
                properties=[],  # No properties are needed as the tool takes no arguments
            ),
        ),
        function=get_user_info,
    )
    return [get_user_info_tool]   

### ### ###

def build_system_prompt_with_tools(native_tools: bool, extra_tools: list = None) -> str:
    """Dynamically builds a system prompt listing all available tools."""
    prompt_lines = ["You have access to the following tools; use them when appropriate to answer the user's request."]
    # Manually add the description for the native Rust tool
    if native_tools:
        prompt_lines.append("- get_current_time: Get the current time.")
    # Dynamically add descriptions for any Python tools
    if extra_tools:
        for tool in extra_tools:
            prompt_lines.append(f"- {tool.definition.name}: {tool.definition.description}")

    return "\n".join(prompt_lines)

TOOLER_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + "\n" + 
    build_system_prompt_with_tools(
        native_tools=True, 
        extra_tools=tool_defs()
    )
)

### ### ###

def tooler_chat():
    """
    Runs an interactive chat session with both native Rust and dynamic Python tools.
    """
    print(
        "--- DYNAMIC TOOL TEST ---\n\n"
        "Ask 'what time is it?' for the Rust tool, or 'My name is Bob and I am 42' for the Python tool.\n"
    )

    chat_wrap(
        model_name=MODEL_NAME,
        system_prompt=TOOLER_SYSTEM_PROMPT,
        native_tools=True,
        extra_tools=tool_defs(),
        debug_out=True,
    )









DATA_ITEMS_TO_SORT = [
    "Apple iPhone 15",
    "Sony WH-1000XM5 Headphones",
    "Dell XPS 15 Laptop",
    "Apple Watch Series 9",
    "Logitech MX Master 3 Mouse"
]

DATA_ITEMS_FILE_PATH = os.path.join(os.path.dirname(__file__), "sample_data.json")
DATA_ITEMS_SORTED_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SORTING_INSTRUCTIONS = SortingInstructions(
    data_item_name="product_name",
    data_profile_description="A list of consumer electronics.",
    item_sorting_guidelines=[
        "Sort items into broad categories like 'audio', 'computing', 'mobile', etc.",
        "If an item fits multiple categories, choose the most specific one."
    ],
    # We can provide some starting categories, or let the AI generate them.
    provided_categories=["audio", "computing", "mobile", "wearable", "accessory"]
)

def data_sorter():
    """
    Take a file path as input, and sort the example data into that file.
    """
    print("\n--- SORTER TEST ---")
    
    # 1. Initialize the input and output paths.
    input_file, output_dir = init_data(
        DATA_ITEMS_FILE_PATH, 
        DATA_ITEMS_SORTED_OUTPUT_DIR
    )
    if input_file and output_dir:
        # 2. Run the sorter. You can switch between sorting from a file or a list.
        
        # Option 1: Sort from the file path.
        sort_data(
            input_file=input_file,
            output_dir=output_dir,
            instructions=SORTING_INSTRUCTIONS,
            debug_out=False
        )
        
        # Option 2: Sort from the list defined in this file.
        # sort_data(
        #     data_items=DATA_ITEMS_TO_SORT,
        #     output_dir=output_dir,
        #     sorting_instructions=SORTING_INSTRUCTIONS,
        #     debug_out=False
        # )









def tester_chat():
    """
    Runs an interactive chat session using the Rust-powered Chat session manager.
    """
    print(
        "--- MY TEST ---" + "\n\n" +
        "Provide a sentence with a name and age for the agent to extract." + "\n"
    ); chat_wrap(
        model_name=MODEL_NAME,
        # tools=
        # debug_out=True
    )

if __name__ == "__main__":
    # Choose which test to run.

    # tester_chat()
    # basics_chat()
    # schema_chat()
    # data_sorter()

    tooler_chat()