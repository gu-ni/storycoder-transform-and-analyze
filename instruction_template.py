INSTRUCTION_LCB = """Please transform the coding problem into a narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use alphabetical symbols for quantity-related variables (such as N, M, K), and you may also use mathematical expressions (such as 10^5) when describing their constraints. However, all other variables must **not** be represented using symbols (such as ≤, ≥, =, or variable names); instead, describe them **indirectly through natural language only.**
- Build the story in **any genre or setting**—such as fantasy, sci-fi, historical, modern, dystopian, mystery, or slice-of-life—and express each mathematical rule through **world-specific logic** such as social norms, systems, behaviors, or relationships.
- You must include and **accurately reflect all original constraints and goals**, converting them into **clear symbolic analogies** within the narrative.
- Clearly convey that the goal is not just to meet the conditions, but to do so **as fully or efficiently as possible** within the world’s logic.
- Use **rich language** to build the world, but ensure that each rule remains **logically clear and inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.
- You **must present the input and output format** as part of the story's narrative.
- Conclude the story by reframing all original sample inputs, outputs, and their explanations in the context of the narrative world.

The story should be structured into **six paragraphs at most**, and follow this flow:

**Background → Rules and Problem Setting → Task Explanation → Examples and Closing**

The coding problem is as follows:

"""

INSTRUCTION_HUMANEVAL = """Please transform the coding problem into a narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use alphabetical symbols for quantity-related variables (such as n, m, k). However, all other variables must **not** be represented using symbols (such as ≤, ≥, =, or variable names); instead, describe them **indirectly through natural language only.**
- Build the story in given genre and express each mathematical rule through **world-specific logic** such as social norms, systems, behaviors, or relationships.
- You must include and **accurately reflect all original constraints and goals**, converting them into **clear symbolic analogies** within the narrative.
- Clearly convey that the goal is not just to meet the conditions, but to do so **as fully or efficiently as possible** within the world’s logic.
- Use **rich language** to build the world, but ensure that each rule remains **logically clear and inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.
- You **must present the input and output format** as part of the story's narrative. If there are multiple inputs, each input is provided on a separate line. If the input or output has multiple lines, your story must reflect this format accurately. Avoid vague phrases like “followed by”—instead, use clear terms like “on the next line” or “on the same line.”
- Conclude the story by reframing all original sample inputs, outputs, and their explanations in the context of the narrative world.
- Do **not** include any intermediate steps that could assist in solving the problem. You **must use only the information explicitly stated in the original problem.**
- Write only what is requested.

The story should be structured into **six paragraphs at most**, and follow this flow:

**Background → Rules and Problem Setting → Task Explanation → Examples and Closing**

Use the following genre: {GENRE}

The coding problem is as follows:

"""


INSTRUCTION_CODEFORCES = """Please transform the coding problem into a narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use alphabetical symbols for quantity-related variables (such as N, M, K), and you may also use mathematical expressions (such as 10^5) when describing their constraints. However, all other variables must **not** be represented using symbols (such as ≤, ≥, =, or variable names); instead, describe them **indirectly through natural language only.**
- Build the story in given genre and express each mathematical rule through **world-specific logic** such as social norms, systems, behaviors, or relationships.
- You must include and **accurately reflect all original constraints and goals**, converting them into **clear symbolic analogies** within the narrative.
- Clearly convey that the goal is not just to meet the conditions, but to do so **as fully or efficiently as possible** within the world’s logic.
- Use **rich language** to build the world, but ensure that each rule remains **logically clear and inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.
- You **must present the input and output format** as part of the story's narrative. If the input or output has multiple lines, your story must reflect this format accurately. Avoid vague phrases like “followed by”—instead, use clear terms like “on the next line” or “on the same line.”
- Conclude the story by reframing all original sample inputs, outputs, and their explanations in the context of the narrative world.

The story should be structured into **six paragraphs at most**, and follow this flow:

**Background → Rules and Problem Setting → Task Explanation → Examples and Closing**

Use the following genre: {GENRE}

The coding problem is as follows:

"""



genres = [
    "Slice of Life School Diary",
    "Post-Apocalyptic Survival Log",
    "Corporate Espionage Thriller",
    "Mythological Hero’s Trial",
    "Time Travel Regulation Protocols",
    "Dream Architect Simulator",
    "Urban Legend Investigator Log",
    "Courtroom Logic Drama",
    "Runestone Puzzle Trials",
    "Space Opera Colony Management",
    "Heist Planning Manual",
    "Ancient Archive Puzzlekeeper",
    "Social Network Popularity Simulator",
    "Genetic Algorithm Lab Notes",
    "Historical Battlefield Logistics",
    "Fantasy Inn Resource Ledger",
    "Mystery Puzzle in Locked Mansion",
    "Underground Hacker’s Terminal Log",
    "Political Simulation RPG",
    "Kingdom Census Ledger",
    "Collaborative Task Scheduling Center",
    "Toy Factory Automation Blueprint",
    "Haunted Library Lexicon Rules",
    "E-Sports Tournament Simulation",
    "Shipwrecked Island Survival Council",
    "Chronicles of the Shifting Labyrinth",
    "Space-Time Puzzle Labyrinth",
    "Lost Civilization Number Rituals",
    "Parallel Universe Synchronization Log",
    "Board Game Rulebook Translation",
    "Carnival Game Engineering Log",
    "Train Station Announcement System",
    "Magical Candy Factory Recipes",
    "Mechanical Puppet Theatre Scripts",
    "Floating Market Merchant Ledger",
    "Arcane Academy Examination",
    "Midnight Radio Broadcast Archive",
    "Monster Evolution Guide",
    "Witch’s Alchemy Book",
    "Abandoned Theme Park Blueprint",
    "Citywide Lantern Festival Logbook",
    "Alien Zoo Containment Manual",
    "Entertainment Event Flow Designer",
    "Tea House Operations Manager",
    "Museum Night Guard Report",
    "Postcard Routing Puzzle",
    "Retro Toy Catalog Compiler",
    "Clockmaker’s Routine Notebook",
    "Ecosystem Simulation Console",
    "Festival Parade Queue Directive"
]



"""
Please summarize the following text, which is a coding problem written in narrative story format. The goal is to make the text more concise while preserving the core explanation of the problem so that a reader can still produce the correct answer. 

### Guidelines for Summarization:

- Remove unnecessary parts of the narrative and retain only the essential elements required to understand the problem and write the correct solution. However, **the original narrative story format must be preserved.**
- You may keep the alphabetical symbols for quantity-related variables (such as N, M, K), and you may also use mathematical expressions (such as 10^5) when describing their constraints. However, all other variables must **not** be represented using symbols (such as ≤, ≥, =, or variable names); instead, **keep them in natural language only.**

The coding problem in narrative story format is as follows:

"""

SHORT_INSTRUCTION_CODEFORCES = """Please transform the coding problem into a brief narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use alphabetical symbols for quantity-related variables (such as N, M, K), and you may also use mathematical expressions (such as 10^5) when describing their constraints. However, all other variables must **not** be represented using symbols (such as ≤, ≥, =, or variable names); instead, describe them **indirectly through natural language only.** Keep these descriptions **brief and unambiguous.**
- Build the story in given genre and express each mathematical rule through **world-specific logic** such as social norms, systems, behaviors, or relationships. Avoid unnecessary elaboration and keep the world-building concise.
- You must include and **accurately reflect all original constraints and goals**, converting them into **clear symbolic analogies** within the narrative.
- Clearly convey that the goal is not just to meet the conditions, but to do so **as fully or efficiently as possible** within the world’s logic.
- Use **rich language** to build the world, but ensure that each rule remains **logically clear and inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.
- You **must present the input and output format** as part of the story's narrative. If the input or output has multiple lines, your story must reflect this format accurately. Avoid vague phrases like “followed by”—instead, use clear terms like “on the next line” or “on the same line.”
- Conclude the story by reframing all original sample inputs, outputs, and their explanations in the context of the narrative world.
- Generate only what is requested.

The story should be short, concise, and structured into **three paragraphs at most**, and follow this flow:

**Rules and Problem Setting → Task Explanation → Examples and Closing**

Use the following genre: {GENRE}

The coding problem is as follows:

"""


SHORT_INSTRUCTION_LCB = """Please transform the coding problem into a brief narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use alphabetical symbols for quantity-related variables (such as N, M, K), and you may also use mathematical expressions (such as 10^5) when describing their constraints. However, all other variables must **not** be represented using symbols (such as ≤, ≥, =, or variable names); instead, describe them **indirectly through natural language only.** Keep these descriptions **brief and unambiguous.**
- Build the story in given genre and express each mathematical rule through **world-specific logic** such as social norms, systems, behaviors, or relationships. Avoid unnecessary elaboration and keep the world-building concise.
- You must include and **accurately reflect all original constraints and goals**, converting them into **clear symbolic analogies** within the narrative.
- Clearly convey that the goal is not just to meet the conditions, but to do so **as fully or efficiently as possible** within the world’s logic.
- Use **rich language** to build the world, but ensure that each rule remains **logically clear and inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.
- You **must present the input and output format** as part of the story's narrative. If the input or output has multiple lines, your story must reflect this format accurately. Avoid vague phrases like “followed by”—instead, use clear terms like “on the next line” or “on the same line.”
- Conclude the story by reframing all original sample inputs, outputs, and their explanations in the context of the narrative world.
- Write only what is requested.

The story should be short, concise, and structured into **three paragraphs at most**, and follow this flow:

**Rules and Problem Setting → Task Explanation → Examples and Closing**

Use the following genre: {GENRE}

The coding problem is as follows:

"""





INSTRUCTION_INCLUDING_HINTS = """Please transform the coding problem into a narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use alphabetical symbols for quantity-related variables (such as n, m, k). All other variables must be described **indirectly and descriptively** using natural language, but in a way that their **functional role and constraints are logically unambiguous**.

- Build the story expressing each mathematical rule and constraint through **world-specific logic** such as social norms, systems, behaviors, or relationships.

- You must include and **accurately reflect all original constraints and goals**, converting them into **clear, symbolic analogies** within the narrative that maintain their original logical precision.

- Clearly convey that the goal is not just to meet the conditions, but to do so **as fully or efficiently as possible** within the world’s logic.

- Use rich language to build the world, but ensure that each rule remains **logically clear and directly inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.

- You **must present the input and output format** as part of the story's narrative. However, ensure that the format is described in a **structurally precise and unambiguous manner** that clearly indicates the number of lines, data types (e.g., integer, string), and their order.

- Conclude the story by reframing all original sample inputs, outputs, and their explanations in the context of the narrative world.

- **Include hints and intermediate steps to guide the solution.** Provide additional context, breakdown the problem into smaller logical components, and offer helpful suggestions that could lead to an efficient solution. The goal is to not only present the problem but also to illuminate a clear path toward solving it.

- Write only what is requested.

- The story should be structured into **six paragraphs at most**, and follow this flow:
    - **Background** → **Rules and Problem Setting** → **Task Explanation** → **Examples and Closing**

- Write in multiple paragraphs, but do not use subheadings.

The coding problem is as follows:

"""


INSTRUCTION_CLARIFY = """Please transform the coding problem into a narrative story format using the following guidelines.

### Guidelines for Narrative Conversion:

- You may use alphabetical symbols for quantity-related variables (such as n, m, k). All other variables must be described **indirectly and descriptively** using natural language, but in a way that their **functional role and constraints are logically unambiguous**.

- When transforming into a narrative, choose one genre from the variety. Build the story expressing each mathematical rule and constraint through **world-specific logic** such as social norms, systems, behaviors, or relationships.

- You must include and **accurately reflect all original constraints and goals**, converting them into **clear, symbolic analogies** within the narrative that maintain their original logical precision.

- Clearly convey that the goal is not just to meet the conditions, but to do so **as fully or efficiently as possible** within the world’s logic.

- Use rich language to build the world, but ensure that each rule remains **logically clear and directly inferable** to the reader. Don't get too caught up in narrative descriptions—focus on clearly explaining the problem as well.

- You **must present the input and output format** as part of the story's narrative. However, ensure that the format is described in a **structurally precise and unambiguous manner** that clearly indicates the number of lines, data types (e.g., integer, string), and their order.

- Conclude the story by reframing all original sample inputs, outputs, and their explanations in the context of the narrative world.

- **You may clarify key conditions to avoid misunderstanding.** The aim is not to suggest a solution, but to ensure the rules are logically clear, and you should avoid offering any potentially misleading or speculative directions.

- The length of the narrative may adapt to the difficulty of the problem: shorter for easy tasks, and longer for medium or hard tasks.

- The story should generally be written in 3 to 6 paragraphs, depending on the complexity of the problem, and follow this flow:
    - **Background** → **Rules and Problem Setting** → **Task Explanation** → **Examples and Closing**

- Write only what is requested.


### The coding problem is as follows:

"""


GEMINI_PARAPHRASE = """Paraphrase the following coding problem while keeping the meaning, constraints, input/output format, and sample cases exactly the same.
You may rephrase the story or wording, but do not alter anything that could change the solution.
The paraphrased question must still produce the same answers as the original.
**Do not attempt to solve the problem or provide any code. Your task is only to paraphrase the question as specified.**
Write only what is requested. The coding problem is as follows:

"""


INSTRUCTION_THREE_COMPONENTS = """Please transform the coding problem into a narrative story using the following guidelines.

### Guidelines for Narrative Conversion:

The narrative must be strictly divided into the following three components:

- Task Overview: Describe the background and objective of the problem in a story-like manner, ensuring the original goal is fully preserved.
- Constraints: Include input sizes, value ranges, special conditions, the essential operational mechanisms of the problem, and, if present in the original statement, any time or efficiency requirements. These should be expressed as natural rules of the narrative world.
- Example Input/Output: Reframe the given input/output examples as narrative situations, but present them clearly and unambiguously, preserving the original format and order.

When combined, the three components must fully and accurately reconstruct the original problem.  
You must write exactly three components, each beginning with the exact headers (- Task Overview, - Constraints, - Example Input/Output), and you must not include any other text outside of these components.

### The coding problem is as follows:

"""


INSTRUCTION_THREE_COMPONENTS_ALGORITHM = """Please transform the coding problem into a narrative story using the following guidelines.

### Guidelines for Narrative Conversion:

Before writing the narrative, you must complete two preliminary steps:

1. Review the major categories of coding test algorithms:
   - Graph Algorithms
   - Dynamic Programming
   - Greedy Algorithms
   - Sorting and Searching
   - String Algorithms
   - Data Structures
   - Mathematics and Number Theory
   - Simulation and Implementation

2. Decide which algorithm category the given problem most closely belongs to.
   Then, select a **narrative genre** that naturally aligns with the chosen algorithm.

### Output Format:

You must write the output in the **exact following order** with the specified headers:

- Algorithm Category: (one of the categories above)

- Narrative Genre: (a fitting genre of your choice)

- Task Overview: Describe the background and objective of the problem in a clear, narrative-inspired manner. The selected algorithm should be introduced naturally here, with its logic explained as part of the setting or scenario.

- Constraints: State input sizes, value ranges, conditions, and key operational rules. If efficiency or time limits exist, express them as natural constraints. The chosen algorithm should also shape these rules.

- Example Input/Output: Reframe the examples as part of the scenario’s flow. Present them as clear, contextual situations.

The narrative must include all essential parts of the original problem, ensuring no constraints, goals, or examples are omitted.
Do not include any other text outside these five sections.
**Do not attempt to solve the problem or provide any code. Your task is only to transform the problem statement into the narrative format as specified.**


### The coding problem is as follows:

"""


INSTRUCTION_THREE_COMPONENTS_ALGORITHM_GIVEN_GENRE = """Please transform the coding problem into a narrative story using the following guidelines.

### Guidelines for Narrative Conversion:

You must write in the {GENRE} style.

Before writing the narrative, you will review the major categories of coding test algorithms:
   - Graph Algorithms
   - Dynamic Programming
   - Greedy Algorithms
   - Sorting and Searching
   - String Algorithms
   - Data Structures
   - Mathematics and Number Theory
   - Simulation and Implementation

### Output Format:

You must write the output in the **exact following order** with the specified headers:

- Algorithm Category: (one of the categories above)

- Task Overview: Describe the background and objective of the problem in a clear, narrative-inspired manner. The selected algorithm should be introduced naturally here, with its logic explained as part of the setting or scenario.

- Constraints: State input sizes, value ranges, conditions, and key operational rules. If efficiency or time limits exist, express them as natural constraints. The chosen algorithm should also shape these rules.

- Example Input/Output: Reframe the examples as part of the scenario’s flow. Present them as clear, contextual situations.

The narrative must include all essential parts of the original problem, ensuring no constraints, goals, or examples are omitted.
Do not include any other text outside these four sections.
**Do not attempt to solve the problem or provide any code. Your task is only to transform the problem statement into the narrative format as specified.**

You must write the narrative in the following genre: {GENRE}


### The coding problem is as follows:

"""


mismatch_genre = [
    "Billboard Advertisement for a Toothbrush",
    "Court Transcript of an Extortion Case",
    "Radio Weather Forecast",
    "Hospital Intake Form",
    "Funeral Service Program",
    "Memorial Tribute Writing",
    "Heavy Machinery Operator License",
    "Obituary Column",
    "Medical Prescription Form",
    "Model Agency Contract",
    "Personal Information Consent Form",
    "Insurance Claim Form",
    "Visa Application Form",
    "Tax Return Form",
    "Military Service Exemption Certificate",
    "Divorce Decree",
    "Bank Loan Agreement",
    "Eulogy",
    "Gravestone Inscription",
    "Condolence Letter",
]


SOT_INSTRUCTION = """You will perform two roles.

Role A (Internal Reasoning Only): You are an explorer who wants to identify and collect different related and specialized subject areas to clarify the question. Your goal is to narrow down the question and provide relevant areas of knowledge and experience you have that help clarify the question. Do NOT output anything from this role. Only use the result internally.

Role B (Final Output): You are an expert in narrative-based explanations for science communication. Your goal is to clarify the question in a narrative way through the interconnected information from Role A, to enable a non-expert to comprehend the question in a more coherent and contextually rich manner. Your goal is **only to clarify the question**, not answer it. Make sure to use all of these narrative techniques when clarifying the question through the interconnected information: Progressive Disclosure, Branching, Analogy, Analogical Reasoning, and Metaphor. This is the ONLY role that produces visible output.

Output ONLY the result of Role B. The question is as follows:

"""