import json
from component_manager import Component_Manager

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, taskInfo):
    # Your code here
    return answer
"""
}

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs (DROP), which assesses the ability to perform discrete reasoning and comprehend detailed information across multiple paragraphs.

## An example question from DROP:

You will be asked to read a passage and answer a question.
Passage:
Non-nationals make up more than half of the population of Bahrain, with immigrants making up about 55% of the overall population.  Of those, the vast majority come from South and Southeast Asia: according to various media reports and government statistics dated between 2005-2009 roughly 290,000 Indians, 125,000 Bangladeshis, 45,000 Pakistanis, 45,000 Filipinos, and 8,000 Indonesians.\nQuestion: What two nationalities had the same number of people living in Bahrain between 2005-2009?
Answer [Not Given]:
Pakistanis and Filipinos 


# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for LLM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

# Description of the role for the LLM
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    \"""
    Function to get JSON response from GPT model.
    
    Args:
    - msg (str): The user message.
    - model (str): The model to use.
    - system_message (str): The system message.
    - temperature (float): Sampling temperature.
    
    Returns:
    - dict: The JSON response.
    \"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=1024,
        stop=None,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    return json_dict

class LLMAgentBase:
    \"""
    Base class for an LLM agent.
    
    Attributes:
    - output_fields (list): Fields expected in the output.
    - agent_name (str): Name of the agent.
    - role (str): Role description for the agent.
    - model (str): Model to be used. (option. Keep it default.)
    - temperature (float): Sampling temperature.
    - id (str): Unique identifier for the agent instance.
    \"""

    def __init__(self, output_fields: list, agent_name: str, role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = temperature
        self.id = random_id()
    
    def generate_prompt(self, input_infos, instruction) -> str:
        \"""
        Generates a prompt for the LLM.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        
        Returns:
        - tuple: System prompt and user prompt.

        An example of a generated prompt:
        ""
        You are a helpful assistant.
        
        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will be given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought Agent hkFo (yourself):
        ...
        
        ### code #1 by Chain-of-Thought Agent hkFo (yourself):
        ...

        ### answer by Chain-of-Thought Agent hkFo's code evaluator:...


        # Instruction: 
        Please think step by step and then solve the task by writing the code.
        ""
        \"""
        output_fields_and_description = {key: f"Your {key}." if not 'answer' in key else f"Your {key}. Return ONLY the alphabet choice, i.e. A or B or C or D." for key in self.output_fields}
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += ' (yourself)'
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx+1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt 

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        \"""
        Queries the LLM with provided input information and instruction.
        
        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.
        
        Returns:
        - output_infos (list[Info]): Output information.
        \"""
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)

        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"
    
    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        # Note:
        # The output of the LLM is a list of Info. If you are only querying one output, you should access it with [0].
        # It is a good practice to always include 'thinking' in the output.
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    def forward(self, taskInfo) -> Union[Info, str]:
        \"""
        Placeholder method for processing task information.
        
        Args:
        - taskInfo (Info): Task information.
        
        Returns:
        - Answer (Union[Info, str]): Your FINAL Answer. Return either a namedtuple Info or a string of answers.
        \"""
        pass
```
# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture. 
Finally, the last key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. 
Also, it could be helpful to set the LLM’s role and temperature to further control the LLM’s response. Note that the LLMAgentBase() will automatically parse the output and return a list of “Infos”. You can get the content by Infos.content. 
DO NOT FORGET the taskInfo input to LLM if you think it is needed, otherwise LLM will not know about the task.

## WRONG Implementation examples:
Here are some mistakes you may make:

1. This is WRONG: ```
feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
```
It is wrong to use "Info('feedback', 'Critic Agent', thinking, 0)". The returned "feedback" from LLMAgentBase is already Info.

2. This is WRONG: ```
# Debugging: Log the generated answer
print('Generated Answer:', ...)
feedback_info = verifier_agent([taskInfo, Info('feedback', 'Critic Agent', thinking, 0)], verification_instruction)
if len(feedback_info) < 3:  # Check if feedback_info has enough elements
    return 'Error: Feedback info incomplete'
```
First, the len(feedback_info) will not work.
Second, you should never return an error message. You should always return the best answer you can get.
Third, you should never print anything in the code.
Lastly, again, DO NOT CREATE Info object by yourself.

3. This is WRONG: ```
all_thinking = []
all_answers = []
for agent, role in zip(agents, roles):
    outputs = agent([taskInfo], independent_reasoning_instruction.format(role=role))
    all_thinking.append(outputs[0].content)
    all_answers.append(outputs[1].content)

# Aggregate the reasoning paths and answers
aggregated_thinking = '\n'.join(all_thinking)
aggregated_answers = '\n'.join(all_answers)
```
You SHOULD NOT extract the content from the Info object by yourself. You should use the Info object directly. If you want to aggregate the content, you should just put those Info objects into a list and then use the list as input to the next LLM agent.

4. This is WRONG: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
response_infos = reasoning_agent([taskInfo] + ..., reasoning_instruction)
    
# Extract the final answer from the response_infos
for info in response_infos:
    if info.name == 'final_answer':
        return info
# Fallback if no answer is found
return Info('answer', 'Final Decision Agent', 'No answer generated.', 0)
```
You should not extract the final answer by yourself. You SHOULD directly return the answer Info. Also, you should always return the best answer you can get.
CORRECT example: ```
reasoning_agent = LLMAgentBase(['thinking', 'answer'], 'Reasoning Agent')
thinking, answer = reasoning_agent([taskInfo] + ..., reasoning_instruction)
return answer
```

# Your task
You are deeply familiar with prompting techniques and the agent works from the literature. Your goal is to maximize the specified performance metrics by proposing interestingly new agents.
Observe the discovered agents carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative when thinking about the next interesting agent to try. You are encouraged to draw inspiration from related agent papers or academic papers from other research areas.
Use the knowledge from the archive and inspiration from academic literature to propose the next interesting agentic system design.
THINK OUTSIDE THE BOX.
"""

# Component_Evaluation_prompt = """
# """

Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""


def get_init_archive():
    return Component_Manager.list_components()


def get_system_prompt():
    return system_prompt

# 准备废弃，因为对此进行更新
def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2



def get_valid_probability(min_p, max_p, N=128):
    from scipy.stats import beta
    """
        计算在给定观测数据下，组件的准确率大于基线准确率的概率，如此则可得到认为组件确实有效的近似概率
        Args:
            min_p: 95%的最小准确率
            max_p: 95%的最大准确率
            N: 总观察次数
        Returns:
            prob: 准确率大于基线准确率的概率
    """
    min_p = min_p / 100.0
    max_p = max_p / 100.0
    k = (min_p * N + max_p * N) / 2
    alpha = k + 1
    beta_param = N - k + 1
    base_p = 0.6
    prob = 1 - beta.cdf(base_p, alpha, beta_param)
    return prob

current_index = 0

def get_component_evaluation_prompt():
    current_components = Component_Manager.list_components()

    for component in current_components:
        component_execute_time = component['execute_time']
        component_min_valid_prob = component['min_valid_prob']
        component_max_valid_prob = component['max_valid_prob']
        valid_prob = get_valid_probability(component_min_valid_prob, component_max_valid_prob)
        # print(f"组件{component['name']}的有效率为{valid_prob:.3f}")
        score = (component_execute_time * valid_prob - (1 - valid_prob)) / component_execute_time
        if score < 0:
            score = 0
        # print(f"组件{component['name']}的得分为{score:.3f}")
        component['score'] = score
    
    Component_Manager.save_components(f"./components_middle_results/components_{current_index}.py")

    # 按照score排序
    current_components.sort(key=lambda x: x['score'], reverse=True)
    # 取前10个组件
    current_components = current_components[:5]
    
    temp = "当前成功入选组件为: \n"
    for item in current_components:
        temp += f"{item['name']}, 得分为 {item['score']:.3f}, 执行时间为 {item['execute_time']:.3f}s\n"
    print(temp)

    # 拼装成一个新的prompt
    component_str = ",\n".join([json.dumps(sol) for sol in current_components])
    component_str = f"[{component_str}]"
    prompt = base.replace("[ARCHIVE]", component_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    return prompt

