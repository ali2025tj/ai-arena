import os
import re
import asyncio
import time
import nest_asyncio
import openai
from openai import AsyncOpenAI
import google.generativeai as genai

# --- SETUP & COMPATIBILITY ---
# Apply patch for Jupyter/Async loops
nest_asyncio.apply()

# Initialize Clients
client = AsyncOpenAI()

# Get Gemini Key
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not gemini_api_key:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables.")
    print("Please set it using: export GEMINI_API_KEY='your_key'")
else:
    genai.configure(api_key=gemini_api_key)

# The Judge and Generator use a stable, smart model
INFRA_MODEL = "gpt-4-turbo" 

# --- UI COLORS ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def get_clean_name(model_slug: str) -> str:
    return model_slug.replace("-", " ").title().replace(" ", "-")

# --- INSTRUCTIONS ---

SOLVER_INSTRUCTIONS = """You are an expert math tutor.
IMPORTANT: You are being evaluated on the CLARITY and DEPTH of your reasoning.
Format your output EXACTLY as follows:

Reasoning:
[Provide a detailed, step-by-step explanation. Explain the 'Why'.]

Final Answer:
[State the final numerical answer or set of values ONLY here.]
"""

JUDGE_TEMPLATE = """You are a Strict Math Judge evaluating a duel between {name_1} and {name_2}.

**JUDGING RULES:**
1. **Accuracy:** Who got the correct answer? (Check for equivalent fractions/sets).
2. **Reasoning:** Who explained it better? 
3. **Speed:** IGNORE SPEED unless it is a perfect tie.

**VERDICT FORMAT:**
If one is clearly better, conclude with: "The winner of the match is: [Name]"
If they are equal in accuracy and reasoning, conclude with: "The winner of the match is: Tie"
"""

GENERATOR_INSTRUCTIONS = """Create a difficult math problem (Algebra, Logic, or Probability). 
Ensure the problem has a unique, verifiable solution.
Format:
Problem Statement: [Prob]
Final Answer: [Ans]"""

# --- CLASSES ---

class Agent:
    def __init__(self, name: str, model: str, instructions: str):
        self.name = name
        self.model = model
        self.instructions = instructions

    async def run(self, input_text: str) -> str:
        try:
            if self.model.startswith('gpt-') or self.model.startswith('o1-'):
                messages = [{"role": "system", "content": self.instructions}, {"role": "user", "content": input_text}]
                response = await client.chat.completions.create(model=self.model, messages=messages, temperature=0.7)
                return response.choices[0].message.content.strip()
            elif 'gemini' in self.model:
                model_instance = genai.GenerativeModel(self.model)
                full_prompt = f"{self.instructions}\n\nUser Input:\n{input_text}"
                response = await model_instance.generate_content_async(full_prompt)
                return response.text.strip()
            else:
                return f"Error: Model '{self.model}' is not supported."
        except Exception as e:
            return f"Error running agent {self.name}: {e}"

class Runner:
    @staticmethod
    async def run(starting_agent: Agent, input_text: str):
        output = await starting_agent.run(input_text=input_text)
        return {"final_output": output}

# --- PARSING LOGIC ---

def parse_math_problem_output(text: str) -> dict:
    try:
        problem_statement = re.search(r"Problem Statement:(.*?)(Final Answer:|$)", text, re.DOTALL | re.IGNORECASE).group(1).strip()
        final_answer_text = re.search(r"Final Answer:(.*)", text, re.DOTALL | re.IGNORECASE).group(1).strip()
        return {"problem_statement": problem_statement, "correct_final_answer": final_answer_text}
    except Exception:
        return {"problem_statement": text, "correct_final_answer": "Unknown"}

def parse_solver_output(text: str) -> dict:
    reasoning, final_answer = "No reasoning provided.", text
    try:
        reasoning_match = re.search(r"Reasoning:(.*?)(Final Answer:|$)", text, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"Final Answer:(.*)", text, re.DOTALL | re.IGNORECASE)
        if reasoning_match: reasoning = reasoning_match.group(1).strip()
        if answer_match: final_answer = answer_match.group(1).strip()
    except Exception:
        pass
    return {"reasoning": reasoning, "final_answer": final_answer}

def extract_winner(verdict_text, name1, name2):
    verdict_lower = verdict_text.lower()
    if "tie" in verdict_lower and "winner" in verdict_lower: return "Tie"
    if f"winner of the match is: {name1.lower()}" in verdict_lower: return name1
    if f"winner of the match is: {name2.lower()}" in verdict_lower: return name2
    if "is: tie" in verdict_lower: return "Tie"
    return None

# --- MAIN LOOP ---

async def main():
    print(f"{bcolors.HEADER}--- ⚔️ WELCOME TO THE DYNAMIC AI ARENA ⚔️ ---{bcolors.ENDC}")
    
    try:
        r_input = input("How many rounds? (Default: 1): ")
        rounds = int(r_input) if r_input.strip() else 1
        
        print(f"\n{bcolors.OKCYAN}Configure Fighter 1:{bcolors.ENDC}")
        m1_input = input("Enter Model ID (default: gpt-5.1): ")
        model_1 = m1_input.strip() if m1_input.strip() else "gpt-5.1"
        name_1 = get_clean_name(model_1)

        print(f"\n{bcolors.OKGREEN}Configure Fighter 2:{bcolors.ENDC}")
        m2_input = input("Enter Model ID (default: gemini-3-pro-preview): ")
        model_2 = m2_input.strip() if m2_input.strip() else "gemini-3-pro-preview"
        name_2 = get_clean_name(model_2)
    except ValueError:
        return

    Math_Problem_Generator = Agent("Generator", INFRA_MODEL, GENERATOR_INSTRUCTIONS)
    current_judge_instructions = JUDGE_TEMPLATE.format(name_1=name_1, name_2=name_2)
    Judge_Agent = Agent("Judge", INFRA_MODEL, current_judge_instructions)
    
    Fighter_1 = Agent(name_1, model_1, SOLVER_INSTRUCTIONS)
    Fighter_2 = Agent(name_2, model_2, SOLVER_INSTRUCTIONS)

    scoreboard = {name_1: 0, name_2: 0, "Tie": 0}

    for i in range(1, rounds + 1):
        print(f"\n{bcolors.HEADER}===== ROUND {i} of {rounds} ====={bcolors.ENDC}")
        print("Generating problem...")
        prob_res = await Runner.run(Math_Problem_Generator, "Generate a hard math problem.")
        prob_data = parse_math_problem_output(prob_res["final_output"])
        print(f"Problem: {bcolors.BOLD}{prob_data['problem_statement']}{bcolors.ENDC}")

        print("Solvers are thinking...")
        task1 = Runner.run(Fighter_1, prob_data['problem_statement'])
        task2 = Runner.run(Fighter_2, prob_data['problem_statement'])
        res1, res2 = await asyncio.gather(task1, task2)

        out1 = parse_solver_output(res1["final_output"])
        out2 = parse_solver_output(res2["final_output"])

        print(f"\n{bcolors.OKCYAN}{name_1}:{bcolors.ENDC} {out1['final_answer']}")
        print(f"{bcolors.OKGREEN}{name_2}:{bcolors.ENDC} {out2['final_answer']}")

        print("\nJudging...")
        judge_input = f"""
        Problem: {prob_data['problem_statement']}
        Generator's Proposed Answer: {prob_data['correct_final_answer']}
        --- {name_1} ---
        Answer: {out1['final_answer']}
        Reasoning: {out1['reasoning']}
        --- {name_2} ---
        Answer: {out2['final_answer']}
        Reasoning: {out2['reasoning']}
        """
        judge_res = await Runner.run(Judge_Agent, judge_input)
        verdict = judge_res["final_output"]
        winner = extract_winner(verdict, name_1, name_2)

        print(f"{bcolors.WARNING}Judge's Verdict:\n{verdict}{bcolors.ENDC}")

        if winner == "Tie":
            scoreboard["Tie"] += 1
            print(f"🤝 Result: {bcolors.BOLD}IT'S A TIE!{bcolors.ENDC}")
        elif winner:
            scoreboard[winner] += 1
            print(f"🏆 Round Winner: {bcolors.BOLD}{winner}{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}Error: Could not determine winner.{bcolors.ENDC}")

        print(f"Score: {name_1}: {scoreboard[name_1]} | {name_2}: {scoreboard[name_2]} | Ties: {scoreboard['Tie']}")
        if i < rounds: await asyncio.sleep(2)

    print(f"\n{bcolors.HEADER}=== FINAL RESULTS ==={bcolors.ENDC}")
    print(f"{name_1}: {scoreboard[name_1]}")
    print(f"{name_2}: {scoreboard[name_2]}")
    print(f"Ties: {scoreboard['Tie']}")

if __name__ == "__main__":
    asyncio.run(main())