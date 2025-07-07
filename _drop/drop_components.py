Components = [
    {
        "thought": """By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.""",
        "name": "Chain-of-Thought",
        "code": """def forward(self, taskInfo):
    # Instruction for the Chain-of-Thought (CoT) approach
    # It is an important practice that allows the LLM to think step by step before solving the task.
    cot_instruction = "Please think step by step and then solve the task."

    # Instantiate a new LLM agent specifically for CoT
    # To allow LLM thinking before answering, we need to set an additional output field 'thinking'.
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Prepare the inputs for the CoT agent
    # The input should be a list of Info, and the first one is often the taskInfo
    cot_agent_inputs = [taskInfo]

    # Get the response from the CoT agent
    thinking, answer = cot_agent(cot_agent_inputs, cot_instruction)

    # Return only the final answer
    return answer
""",
        "execute_time": 21.365790128707886,
        "acc_score": 65.974296875,
        "min_valid_prob": 52.665680683593756,
        "max_valid_prob": 57.021637578125
    },
    {
        "thought": """While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.""",
        "name": "Self-Consistency with Chain-of-Thought",
        "code": """def forward(self, taskInfo):
    # Instruction for step-by-step reasoning
    cot_instruction = "Please think step by step and then solve the task."
    N = 5 # Number of CoT agents

    # Initialize multiple CoT agents with a higher temperature for varied reasoning
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]

    # Instruction for final decision-making based on collected reasoning and answers
    final_decision_instruction = "Given all the above solutions, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    possible_answers = []
    for i in range(N):
        thinking, answer = cot_agents[i]([taskInfo], cot_instruction)
        possible_answers.extend([thinking, answer])

    # Make the final decision based on all generated answers
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer  
""",
        "execute_time": 37.947970151901245,
        "acc_score": 67.92453125,
        "min_valid_prob": 54.78061484375,
        "max_valid_prob": 59.0478616796875
    },
    {
        "thought": """To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.""",
        "name": "Self-Refine (Reflexion)",
        "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Instruction for providing feedback and correcting the answer
    critic_instruction = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output 'True' in 'correct'."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent')
    
    N_max = 5 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # Get feedback and correct status from the critic
        feedback, correct = critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([thinking, answer, feedback])

        # Reflect on previous attempts and refine the answer
        thinking, answer = cot_agent(cot_inputs, cot_reflect_instruction, i + 1)
    return answer
""",
        "execute_time": 50.17393374443054,
        "acc_score": 58.953203125,
        "min_valid_prob": 45.52968603515625,
        "max_valid_prob": 50.024215703124995
    },
    {
        "thought": """By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.""",
        "name": "LLM Debate",
        "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning
    debate_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for debating and updating the solution based on other agents' solutions
    debate_instruction = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    
    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    debate_agents = [LLMAgentBase(['thinking', 'answer'], 'Debate Agent', temperature=0.8, role=role) for role in ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator']]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_instruction = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)

    max_round = 2 # Maximum number of debate rounds
    all_thinking = [[] for _ in range(max_round)]
    all_answer = [[] for _ in range(max_round)]

    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0:
                thinking, answer = debate_agents[i]([taskInfo], debate_initial_instruction)
            else:
                input_infos = [taskInfo] + [all_thinking[r-1][i]] + all_thinking[r-1][:i] + all_thinking[r-1][i+1:]
                thinking, answer = debate_agents[i](input_infos, debate_instruction)
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    
    # Make the final decision based on all debate results and solutions
    thinking, answer = final_decision_agent([taskInfo] + all_thinking[max_round-1] + all_answer[max_round-1], final_decision_instruction)
    return answer
""",
        "execute_time": 41.69117569923401,
        "acc_score": 62.958984375,
        "min_valid_prob": 49.1412484765625,
        "max_valid_prob": 53.7864706640625
    },
    {
        "thought": """Let LLM first think about the principles involved in solving this task which could be helpful. By understanding the underlying principles, the model can better reason through the problem and provide a more accurate solution.""",
        "name": "Step-back Abstraction",
        "code": """def forward(self, taskInfo):
        # Instruction for understanding the principles involved in the task
        principle_instruction = "What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        
        # Instruction for solving the task based on the principles
        cot_instruction = "Given the question and the involved principle behind the question, think step by step and then solve the task."
        
        # Instantiate LLM agents
        principle_agent = LLMAgentBase(['thinking', 'principle'], 'Principle Agent')
        cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')
        
        # Get the principles involved in the task
        thinking, principle = principle_agent([taskInfo], principle_instruction)

        # Use the principles to solve the task
        thinking, answer = cot_agent([taskInfo, thinking, principle], cot_instruction)
        return answer
""",
        "execute_time": 39.704288721084595,
        "acc_score": 61.865468750000005,
        "min_valid_prob": 48.160911015625,
        "max_valid_prob": 52.794059453124994
    },
    {
        "thought": """Similar to Quality-Diversity methods, let LLM generate multiple diverse interesting solutions could help. By encouraging the model to explore different reasoning paths, we can increase the chances of finding the best solution.""",
        "name": "Quality-Diversity",
        "code": """def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for giving diverse answers
    qd_instruction = "Given previous attempts, try to come up with another interesting way to solve the task."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent')

    # Instruction for final decision-making based on collected reasoning and answers
    final_decision_instruction = "Given all the above solutions, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    
    N_max = 3 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    possible_answers = []
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)

    # Add the answer to the list of possible answers
    possible_answers.extend([thinking, answer])

    for i in range(N_max):
        # Reflect on previous attempts and generate another interesting answer
        cot_inputs.extend([thinking, answer])

        # Generate another interesting answer
        thinking, answer = cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])

    # Make the final decision based on all generated answers
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer
""",
        "execute_time": 34.22123408317566,
        "acc_score": 63.424609375,
        "min_valid_prob": 49.23054107421875,
        "max_valid_prob": 54.123507246093745
    },
    {
        "thought": """Similar to Auto-GPT and expert prompting, we can use dynamic control flow in the design to let the agent decide what expert we should use.""",
        "name": "Dynamic Assignment of Roles",
        "code": """def forward(self, taskInfo):
        # Instruction for step-by-step reasoning
        cot_instruction = "Please think step by step and then solve the task."
        expert_agents = [LLMAgentBase(['thinking', 'answer'], 'Expert Agent', role=role) for role in ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator', 'Helpful Assistant']]

        # Instruction for routing the task to the appropriate expert
        routing_instruction = "Given the task, please choose an Expert to answer the question. Choose from: Reading Comprehension Specialist, Logical Reasoning Strategist, and Multidisciplinary Knowledge Integrator."
        routing_agent = LLMAgentBase(['choice'], 'Routing agent')

        # Get the choice of expert to route the task
        choice = routing_agent([taskInfo], routing_instruction)[0]

        if 'specialist' in choice.content.lower():
            expert_id = 0
        elif 'strategist' in choice.content.lower():
            expert_id = 1
        elif 'integrator' in choice.content.lower():
            expert_id = 2
        else:
            expert_id = 3 # Default to helpful assistant

        thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
        return answer
""",
        "execute_time": 46.780869722366333,
        "acc_score": 69.06351562500001,
        "min_valid_prob": 56.0148346484375,
        "max_valid_prob": 60.35005984375
    },
    {
        "thought": """**Insights:**
Combining domain-specific expertise with iterative refinement can enhance performance. By allowing domain experts to refine their answers based on feedback from a verification agent, we can improve the system's reliability and accuracy.

**Overall Idea:**
The system will first use domain-specific experts to generate initial solutions. These answers will be verified and cross-checked by a verification agent. The domain experts will then refine their answers based on feedback from the verification agent. This iterative process will continue until the system converges on a final answer or reaches a maximum number of iterations.

**Implementation:**
Design a forward function that uses domain-specific experts to generate initial solutions, employs a verification agent to provide feedback, and iterates the process until convergence or a maximum number of iterations.""",
        "name": "Domain Expertise with Iterative Refinement",
        "code": """def forward(self, taskInfo):
    # Instruction for experts
    expert_instruction = 'Please think step by step and then solve the task.'
    refinement_instruction = 'Refine your answer based on the feedback provided.'
    
    # Initialize domain-specific experts
    math_agent = LLMAgentBase(['thinking', 'answer'], 'Math Expert')
    logic_agent = LLMAgentBase(['thinking', 'answer'], 'Logical Reasoning Expert')
    knowledge_agent = LLMAgentBase(['thinking', 'answer'], 'Domain Knowledge Expert')

    # Instruction for verification
    verification_instruction = 'Please cross-check the answers provided by different experts and provide feedback.'
    
    # Initialize verification agent
    verification_agent = LLMAgentBase(['feedback'], 'Verification Agent')

    # Maximum number of iterations
    max_iterations = 3

    # Initial solutions from domain-specific experts
    math_thinking, math_answer = math_agent([taskInfo], expert_instruction)
    logic_thinking, logic_answer = logic_agent([taskInfo], expert_instruction)
    knowledge_thinking, knowledge_answer = knowledge_agent([taskInfo], expert_instruction)

    for i in range(max_iterations):
        # Combine answers for verification
        verification_inputs = [taskInfo, math_thinking, math_answer, logic_thinking, logic_answer, knowledge_thinking, knowledge_answer]
        feedback = verification_agent(verification_inputs, verification_instruction)[0]

        # Refine answers based on feedback
        math_thinking, math_answer = math_agent([taskInfo, feedback], refinement_instruction, i)
        logic_thinking, logic_answer = logic_agent([taskInfo, feedback], refinement_instruction, i)
        knowledge_thinking, knowledge_answer = knowledge_agent([taskInfo, feedback], refinement_instruction, i)

    # Final verification to decide the best answer
    verification_inputs = [taskInfo, math_thinking, math_answer, logic_thinking, logic_answer, knowledge_thinking, knowledge_answer]
    final_verification_instruction = 'Please provide the final verified answer based on the expert solutions.'
    final_answer = verification_agent(verification_inputs, final_verification_instruction)[0]

    # Return the final verified answer
    return final_answer
""",
        "execute_time": 73.10436773300171,
        "acc_score": 47.062109375,
        "min_valid_prob": 33.493124804687504,
        "max_valid_prob": 37.823579277343754
    },
    {
        "thought": """**Insights:**
Combining domain-specific expertise with a structured peer review process can enhance performance. By allowing domain experts to critique each other's work and having an Editor Agent synthesize the critiques, we can improve the system's reliability and accuracy.

**Overall Idea:**
The system will first use domain-specific experts to generate initial solutions. These answers will be critiqued by other experts in a peer review process. An Editor Agent will then synthesize the critiques and provide the final answer.

**Implementation:**
Design a forward function that uses domain-specific experts to generate initial solutions, employs other experts to critique these solutions, and synthesizes the critiques to provide the final answer.""",
        "name": "Peer Review Agent",
        "code": """def forward(self, taskInfo):
    # Step 1: Initial Generation
    initial_instruction = "Please think step by step and then solve the task."
    roles = ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator']
    initial_agents = [LLMAgentBase(['thinking', 'answer'], 'Initial Agent', role=role) for role in roles]

    all_initial_thinking = []
    all_initial_answers = []
    for agent in initial_agents:
        thinking, answer = agent([taskInfo], initial_instruction)
        all_initial_thinking.append(thinking)
        all_initial_answers.append(answer)

    # Step 2: Peer Review
    review_instruction = "Given the solutions from other agents, critique their responses. Provide detailed feedback and suggest improvements."
    review_agents = [LLMAgentBase(['feedback'], 'Review Agent', role=role) for role in roles]

    all_feedback = []
    for i, agent in enumerate(review_agents):
        inputs = [taskInfo] + all_initial_thinking + all_initial_answers[:i] + all_initial_answers[i+1:]
        feedback = agent(inputs, review_instruction)[0]
        all_feedback.append(feedback)

    # Step 3: Editor Decision
    editor_instruction = "Given all the feedback and initial answers, synthesize the information and provide the final answer."
    editor_agent = LLMAgentBase(['thinking', 'answer'], 'Editor Agent')

    inputs = [taskInfo] + all_initial_thinking + all_initial_answers + all_feedback
    thinking, answer = editor_agent(inputs, editor_instruction)

    return answer
""",
        "execute_time": 52.96423840522766,
        "acc_score": 56.8053125,
        "min_valid_prob": 42.811543554687496,
        "max_valid_prob": 47.6814737109375
    },
    {
        "thought": """**Insights:**
Leveraging the power of multiple expert agents working together and refining each other's answers iteratively can lead to more accurate final solutions. By enabling collaborative refinement among agents in successive rounds, we can achieve continuous improvement and a more integrated final solution.

**Overall Idea:**
The system will have multiple expert agents provide initial solutions independently. These solutions will then be collaboratively refined by the agents in successive rounds. The final decision agent will synthesize the refined solutions and provide the final answer.

**Implementation:**
Design a forward function that uses expert agents to generate initial solutions, employs collaborative refinement among agents in successive rounds, and synthesizes the refined solutions to provide the final answer.""",
        "name": "Collaborative Refinement Agent",
        "code": """def forward(self, taskInfo):
    # Step 1: Initial Generation
    initial_instruction = 'Please think step by step and then solve the task.'
    roles = ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator']
    initial_agents = [LLMAgentBase(['thinking', 'answer'], 'Initial Agent', role=role) for role in roles]

    all_initial_infos = []
    for agent in initial_agents:
        infos = agent([taskInfo], initial_instruction)
        all_initial_infos.extend(infos)

    # Step 2: Collaborative Refinement
    refinement_instruction = 'Given the initial solutions from other agents, collaboratively refine your solution considering their insights.'
    refinement_agents = [LLMAgentBase(['thinking', 'answer'], 'Refinement Agent', role=role) for role in roles]
    max_rounds = 2

    refined_infos = all_initial_infos
    for _ in range(max_rounds):
        new_infos = []
        for agent in refinement_agents:
            inputs = [taskInfo] + refined_infos
            infos = agent(inputs, refinement_instruction)
            new_infos.extend(infos)
        refined_infos = new_infos

    # Step 3: Final Decision
    final_decision_instruction = 'Given all the refined solutions, reason over them carefully and provide a final answer.'
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    inputs = [taskInfo] + refined_infos
    final_infos = final_decision_agent(inputs, final_decision_instruction)

    # Return the final answer
    return final_infos[-1]
""",
        "execute_time": 64.82552027702332,
        "acc_score": 65.31796875,
        "min_valid_prob": 51.37835583984375,
        "max_valid_prob": 56.167780507812495
    },
    {
        "thought": """**Insights:**
Leveraging the power of intermediate decision-making can enhance the collaborative refinement process. By synthesizing intermediate solutions, we can guide the refinement agents more effectively, ensuring that each step builds on the previous one more coherently.

**Overall Idea:**
The architecture will consist of three stages: Initial Generation, Intermediate Decision-Making, and Final Decision. In the initial stage, multiple expert agents provide their solutions independently. In the intermediate decision-making stage, an agent synthesizes these solutions after each refinement round and provides the synthesized result to guide further refinements. Finally, the final decision agent synthesizes all refined solutions to provide the final answer.""",
        "name": "Intermediate Decision-Making Agent",
        "code": """def forward(self, taskInfo):
    # Step 1: Initial Generation
    initial_instruction = 'Please think step by step and then solve the task.'
    roles = ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator']
    initial_agents = [LLMAgentBase(['thinking', 'answer'], 'Initial Agent', role=role) for role in roles]

    all_initial_infos = []
    for agent in initial_agents:
        infos = agent([taskInfo], initial_instruction)
        all_initial_infos.extend(infos)

    # Step 2: Intermediate Decision-Making and Collaborative Refinement
    refinement_instruction = 'Given the intermediate solutions, collaboratively refine your solution considering their insights.'
    intermediate_decision_instruction = 'Synthesize the intermediate solutions and provide a synthesized result to guide further refinements.'
    refinement_agents = [LLMAgentBase(['thinking', 'answer'], 'Refinement Agent', role=role) for role in roles]
    intermediate_decision_agent = LLMAgentBase(['synthesized_result'], 'Intermediate Decision Agent', temperature=0.1)
    max_rounds = 2

    refined_infos = all_initial_infos
    for _ in range(max_rounds):
        new_infos = []
        for agent in refinement_agents:
            inputs = [taskInfo] + refined_infos
            infos = agent(inputs, refinement_instruction)
            new_infos.extend(infos)
        synthesized_infos = intermediate_decision_agent([taskInfo] + new_infos, intermediate_decision_instruction)
        refined_infos = new_infos + synthesized_infos

    # Step 3: Final Decision
    final_decision_instruction = 'Given all the refined solutions, reason over them carefully and provide a final answer.'
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    inputs = [taskInfo] + refined_infos
    final_infos = final_decision_agent(inputs, final_decision_instruction)

    # Return the final answer
    return final_infos[-1]
""",
        "execute_time": 66.77604651451111,
        "acc_score": 64.078046875,
        "min_valid_prob": 50.436324140625,
        "max_valid_prob": 54.981168828125
    },
    {
        "thought": """**Insights:**
Leveraging a feedback loop and self-assessment can significantly improve the accuracy and robustness of the solutions generated by multiple agents. By incorporating a 'Self-Evaluation' phase, each agent can critically assess its solution and make necessary adjustments before contributing to the final decision. This practice can catch errors and inconsistencies early, improving the overall quality of the final output.

**Overall Idea:**
The architecture will consist of four stages: Initial Generation, Self-Evaluation, Collaborative Refinement, and Final Decision. In the initial stage, multiple expert agents provide their solutions independently. In the self-evaluation stage, each agent critically assesses its solution and makes necessary adjustments. In the collaborative refinement stage, agents improve solutions based on insights from other agents. Finally, the final decision agent synthesizes all refined solutions to provide the final answer.

**Implementation:**
The implementation involves setting up agents for initial generation, self-evaluation, collaborative refinement, and final decision, and orchestrating them in the specified stages.""",
        "name": "Self-Evaluation and Feedback Loop Agent",
        "code": """def forward(self, taskInfo):
    # Step 1: Initial Generation
    initial_instruction = 'Please think step by step and then solve the task.'
    roles = ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator']
    initial_agents = [LLMAgentBase(['thinking', 'answer'], 'Initial Agent', role=role) for role in roles]

    all_initial_infos = []
    for agent in initial_agents:
        infos = agent([taskInfo], initial_instruction)
        all_initial_infos.extend(infos)

    # Step 2: Self-Evaluation
    self_eval_instruction = 'Please critically assess your solution, identify any potential issues, and make necessary adjustments.'
    self_eval_agents = [LLMAgentBase(['self_thinking', 'self_assessed_answer'], 'Self-Evaluation Agent', role=role) for role in roles]

    self_eval_infos = []
    for agent in self_eval_agents:
        infos = agent([taskInfo] + all_initial_infos, self_eval_instruction)
        self_eval_infos.extend(infos)

    # Step 3: Collaborative Refinement
    refinement_instruction = 'Given the self-assessed solutions, collaboratively refine your solution considering insights from other agents.'
    refinement_agents = [LLMAgentBase(['thinking', 'answer'], 'Refinement Agent', role=role) for role in roles]
    max_rounds = 2

    refined_infos = self_eval_infos
    for _ in range(max_rounds):
        new_infos = []
        for agent in refinement_agents:
            inputs = [taskInfo] + refined_infos
            infos = agent(inputs, refinement_instruction)
            new_infos.extend(infos)
        refined_infos = new_infos

    # Step 4: Final Decision
    final_decision_instruction = 'Given all the refined solutions, reason over them carefully and provide a final answer.'
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    inputs = [taskInfo] + refined_infos
    final_infos = final_decision_agent(inputs, final_decision_instruction)

    # Return the final answer
    return final_infos[-1]
""",
        "execute_time": 76.7570877075195,
        "acc_score": 57.854765625,
        "min_valid_prob": 43.93390091796875,
        "max_valid_prob": 48.54420580078125
    },
    {
        "thought": """**Insights:**
Combining iterative refinement with diverse initial solutions can enhance the overall quality and accuracy of the final answer. By simplifying the process and focusing on effective collaboration among agents, the architecture can achieve better performance.

**Overall Idea:**
The architecture will generate diverse initial solutions using multiple CoT agents and then iteratively refine these solutions using expert agents. The final answer will be synthesized from the refined solutions, ensuring a streamlined and effective process.

**Implementation:**
The implementation involves setting up CoT agents for initial generation, expert agents for iterative refinement, and a final decision agent to synthesize the final answer.""",
        "name": "Iterative Refinement Agent",
        "code": """def forward(self, taskInfo):
    # Step 1: Initial Generation with Chain-of-Thought Agents
    initial_instruction = 'Please think step by step and then solve the task.'
    N = 3  # Number of CoT agents for diverse generation
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]
    all_initial_infos = []
    for agent in cot_agents:
        infos = agent([taskInfo], initial_instruction)
        all_initial_infos.extend(infos)

    # Step 2: Iterative Collaborative Refinement
    refinement_instruction = 'Given the diverse initial solutions, collaboratively refine your solution considering insights from other solutions.'
    roles = ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator']
    refinement_agents = [LLMAgentBase(['thinking', 'answer'], 'Refinement Agent', role=role) for role in roles]
    max_rounds = 3
    refined_infos = all_initial_infos
    for _ in range(max_rounds):
        new_infos = []
        for agent in refinement_agents:
            inputs = [taskInfo] + refined_infos
            infos = agent(inputs, refinement_instruction)
            new_infos.extend(infos)
        refined_infos = new_infos

    # Step 3: Final Decision
    final_decision_instruction = 'Given all the refined solutions, reason over them carefully and provide a final answer.'
    final_decision_agent = LLMAgentBase(['thinking', 'answer'], 'Final Decision Agent', temperature=0.1)
    inputs = [taskInfo] + refined_infos
    final_infos = final_decision_agent(inputs, final_decision_instruction)

    # Return the final answer
    return final_infos[-1]
""",
        "execute_time": 67.4188551902771,
        "acc_score": 63.36109375,
        "min_valid_prob": 49.4061458984375,
        "max_valid_prob": 54.18116349609375
    },
    {
        "thought": """**Insights:**
Combining iterative refinement with a dedicated cross-verification stage can enhance the overall quality and accuracy of the final answer. By simplifying the process and focusing on effective collaboration among agents, the architecture can achieve better performance.

**Overall Idea:**
The architecture will generate diverse initial solutions using multiple CoT agents and then iteratively refine these solutions using expert agents. The final synthesized solutions will then be cross-verified by a specialized verification agent to ensure the final answer is accurate and reliable.

**Implementation:**
The implementation involves setting up CoT agents for initial generation, expert agents for iterative refinement, and a final verification agent to cross-verify the refined solutions and provide the final answer.""",
        "name": "Cross-Verification Agent",
        "code": """def forward(self, taskInfo):
    # Step 1: Initial Generation with Chain-of-Thought Agents
    initial_instruction = 'Please think step by step and then solve the task.'
    N = 3  # Number of CoT agents for diverse generation
    cot_agents = [LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', temperature=0.8) for _ in range(N)]
    all_initial_infos = []
    for agent in cot_agents:
        all_initial_infos.extend(agent([taskInfo], initial_instruction))

    # Step 2: Iterative Collaborative Refinement
    refinement_instruction = 'Given the diverse initial solutions, collaboratively refine your solution considering insights from other solutions.'
    roles = ['Reading Comprehension Specialist', 'Logical Reasoning Strategist', 'Multidisciplinary Knowledge Integrator']
    refinement_agents = [LLMAgentBase(['thinking', 'answer'], 'Refinement Agent', role=role) for role in roles]
    max_rounds = 3
    refined_infos = all_initial_infos
    for _ in range(max_rounds):
        new_infos = []
        for agent in refinement_agents:
            inputs = [taskInfo] + refined_infos
            new_infos.extend(agent(inputs, refinement_instruction))
        refined_infos = new_infos

    # Step 3: Cross-Verification
    cross_verification_instruction = 'Given the refined solutions, cross-verify and validate the solutions to ensure the final answer is accurate and reliable.'
    cross_verification_agent = LLMAgentBase(['thinking', 'answer'], 'Verification Agent', temperature=0.1)
    inputs = [taskInfo] + refined_infos
    thinking, answer = cross_verification_agent(inputs, cross_verification_instruction)

    # Return the final answer
    return answer
""",
        "execute_time": 72.75242233276367,
        "acc_score": 62.733359375,
        "min_valid_prob": 49.0866341015625,
        "max_valid_prob": 53.415465703125
    }
]
