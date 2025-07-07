import argparse
import copy
import json
import os
import random
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import backoff
import numpy as np
import openai
from tqdm import tqdm

from dotenv import load_dotenv

from drop_prompt import get_init_archive, get_system_prompt, get_reflexion_prompt, get_component_evaluation_prompt
from component_manager import Component_Manager

# 加载.env文件中的环境变量
load_dotenv()

client = openai.OpenAI(
    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
    api_key=os.getenv('OPENAI_API_KEY')
)

from utils import random_id, bootstrap_confidence_interval, load_drop, drop_metric

# 定义信息结构体，用于存储搜索过程中的各种信息
# name: 信息类型名称, author: 创建者, content: 内容, iteration_idx: 迭代索引
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# 输出格式约束函数，确保LLM按照指定的JSON格式输出
FORMAT_INST = lambda request_keys: f"""Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!\n"""

# 角色定义函数，为LLM设定角色身份
ROLE_DESC = lambda role: f"You are a {role}."

# 系统消息
SYSTEM_MSG = ""

# 调试和模式配置
PRINT_LLM_DEBUG = False  # 是否打印LLM调试信息
SEARCHING_MODE = True    # 当前是否处于搜索模式（True: 搜索, False: 评估）

# 获取GPT模型的JSON响应，包含错误重试机制
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
):
    """
    从GPT模型获取JSON格式的响应
    
    Args:
        msg: 用户消息
        model: 使用的模型名称
        system_message: 系统提示消息
        temperature: 温度参数，控制随机性
    
    Returns:
        dict: 解析后的JSON响应
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, 
        max_tokens=4096, 
        stop=None, 
        response_format={"type": "json_object"}  # 强制JSON格式输出
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict

# 获取GPT模型的反思响应，用于多轮对话
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
):
    """
    从GPT模型获取反思响应，支持多轮对话
    
    Args:
        msg_list: 消息列表，包含完整的对话历史
        model: 使用的模型名称
        temperature: 温度参数，反思时使用较高的随机性
    
    Returns:
        dict: 解析后的JSON响应
    """
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, 
        max_tokens=4096, 
        stop=None, 
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


class LLMAgentBase():
    """
    LLM智能体基础类
    
    这个类封装了与大语言模型交互的基本功能，包括提示生成、
    模型查询和结果处理。每个智能体都有特定的输出字段和角色设定。
    
    Attributes:
        output_fields: 期望的输出字段列表
        agent_name: 智能体名称
        role: 智能体角色描述
        model: 使用的语言模型
        temperature: 温度参数
        id: 唯一标识符
    """

    def __init__(self, output_fields: list, agent_name: str,
                 role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        """
        初始化LLM智能体
        
        Args:
            output_fields: 期望输出的字段列表
            agent_name: 智能体名称
            role: 角色描述，默认为'helpful assistant'
            model: 使用的模型，默认为'gpt-3.5-turbo-0125'
            temperature: 温度参数，默认为0.5
        """
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # 为每个实例分配唯一ID
        self.id = random_id()

    def generate_prompt(self, input_infos, instruction) -> str:
        """
        生成系统提示和用户提示
        
        Args:
            input_infos: 输入信息列表
            instruction: 指令内容
            
        Returns:
            tuple: (系统提示, 用户提示)
        """
        # 构建系统提示
        output_fields_and_description = {
            key: f"Your {key}." if not 'answer' in key else f"Your {key}. Directly answer the question. Keep it very concise." 
            for key in self.output_fields
        }
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(output_fields_and_description)

        # 构建输入信息文本
        input_infos_text = ''
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            # 标记自己生成的内容
            if author == self.__repr__():
                author += ' (yourself)'
            # 根据字段名称和迭代索引格式化内容
            if field_name == 'task':
                input_infos_text += f'# Your Task:\n{content}\n\n'
            elif iteration_idx != -1:
                input_infos_text += f'### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n'
            else:
                input_infos_text += f'### {field_name} by {author}:\n{content}\n\n'

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        """
        查询模型并处理响应
        
        Args:
            input_infos: 输入信息列表
            instruction: 指令内容
            iteration_idx: 迭代索引，默认为-1
            
        Returns:
            list: Info对象列表，包含模型的输出
        """
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = get_json_response_from_gpt(prompt, self.model, system_prompt, self.temperature)
            assert len(response_json) == len(self.output_fields), "not returning enough fields"
        except Exception as e:
            # print(e)
            # 处理上下文长度超限的情况
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError("The context is too long. Please try to design the agent to have shorter context.")
            # 尝试修复缺失字段
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(self.output_fields):
                    response_json[key] = ''
            # 删除多余字段
            for key in copy.deepcopy(list(response_json.keys())):
                if len(response_json) > len(self.output_fields) and not key in self.output_fields:
                    del response_json[key]
        
        # 将响应转换为Info对象列表
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        """返回智能体的字符串表示"""
        return f"{self.agent_name} {self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        """使智能体对象可调用"""
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentSystem():
    """
    智能体系统类
    
    这个类作为智能体系统的容器，用于在运行时动态设置forward方法。
    forward方法在evaluate_forward_fn函数中通过exec动态定义。
    """
    def __init__(self) -> None:
        pass

def save_new_component(next_solution, execute_time, acc_score, lower_bound, upper_bound):
    """
    保存新的组件

    Args:
        next_solution: 新的解决方案
        execute_time: 执行时间
        acc_score: 准确率
    """

    if Component_Manager.find_component_by_name(next_solution['name']):
        Component_Manager.update_component(next_solution['thought'], next_solution['name'], next_solution['code'], execute_time, acc_score, lower_bound, upper_bound)
        # print(f"这里更新组件: {next_solution['name']}, 传入参数为: {execute_time}, {acc_score}, {lower_bound}, {upper_bound}")
    else:
        Component_Manager.add_component(next_solution['thought'], next_solution['name'], next_solution['code'], execute_time, acc_score, lower_bound, upper_bound)
        # print(f"这里保存新的组件: {next_solution['name']}, 传入参数为: {execute_time}, {acc_score}, {lower_bound}, {upper_bound}")

def search(args):
    """
    执行搜索过程
    
    这是系统的核心函数，实现了迭代搜索算法：
    1. 加载或初始化解决方案存档
    2. 评估初始存档中的解决方案
    3. 进行多轮迭代，每轮生成新的解决方案
    4. 对新解决方案进行反思和调试
    5. 评估并保存结果
    
    Args:
        args: 命令行参数对象，包含各种配置项
    """
    # 构建保存搜索结果的文件路径
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    
    # 检查文件是否存在，如果存在则读取，否则初始化
    # if os.path.exists(file_path):
    #     with open(file_path, 'r') as json_file:
    #         archive = json.load(json_file)
    #     # 确定搜索的起始轮次    
    #     if "generation" in archive[-1] and isinstance(archive[-1]['generation'], int):
    #         start = archive[-1]['generation']
    #     else:
    #         start = 0
    # else:
        # archive = get_init_archive()  # 获取初始解决方案存档
        # start = 0
    archive = get_init_archive()  # 获取初始解决方案存档
    start = 0
    # 遍历初始存档中的解决方案，评估尚未评估的解决方案
    for solution in archive:
        # if 'fitness' in solution:
        #     continue # 如果解决方案已有适应度值，则跳过

        solution['generation'] = "initial" # 设置轮次为初始
        print(f"============Initial Archive: {solution['name']}=================")
        try:
            # 记录评估开始时间
            eval_start_time = time.time()
            acc_list = evaluate_forward_fn(args, solution["code"])  # 评估解决方案的准确率列表
            # 记录评估结束时间并计算执行时间
            eval_end_time = time.time()
            eval_execution_time = eval_end_time - eval_start_time

        except Exception as e:
            print("During evaluating initial archive:")
            print(e)
            continue

        fitness_str, lower_bound, upper_bound = bootstrap_confidence_interval(acc_list) # 计算准确率的置信区间
        solution['fitness'] = fitness_str # 设置解决方案的适应度值
        save_new_component(solution, eval_execution_time, np.mean(acc_list), lower_bound, upper_bound) # 保存新的组件

        # 保存评估结果
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)

    # 进行多轮次迭代搜索
    for n in range(start, args.n_generation):
        print(f"============Generation {n + 1}=================")
        system_prompt = get_system_prompt() # 获取系统提示
        # system_prompt, prompt = get_prompt(archive) # 获取基于当前存档的系统提示和搜索提示
        msg_list = [
            {"role": "system", "content": system_prompt},
            # {"role": "user", "content": prompt},
        ]
        try:
            # 获取组件评估，用于挑选合适的组件
            component_evlauation_prompt = get_component_evaluation_prompt()
            # print(component_evlauation_prompt)  # 打印组件评估提示
            msg_list.append({"role": "assistant", "content": component_evlauation_prompt})
            
            # 将msg_list保存到文件中
            msg_list_file_path = os.path.join(args.save_dir, f"{args.expr_name}_generation_{n+1}_msg_list.json")
            os.makedirs(os.path.dirname(msg_list_file_path), exist_ok=True)
            with open(msg_list_file_path, 'w', encoding='utf-8') as f:
                json.dump(msg_list, f, ensure_ascii=False, indent=2)
            # print(f"msg_list已保存到: {msg_list_file_path}")

            # 从模型中获取新的解决方案
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)

            # 获取反思提示，用于改进解决方案
            Reflexion_prompt_1, Reflexion_prompt_2 = get_reflexion_prompt(archive[-1] if n > 0 else None)
            
            # 第一次反思：基于初始生成的解决方案进行反思
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_1})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
            
            # 第二次反思：进一步优化解决方案
            msg_list.append({"role": "assistant", "content": str(next_solution)})
            msg_list.append({"role": "user", "content": Reflexion_prompt_2})
            next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)


        except Exception as e:
            print("During LLM generate new solution:")
            print(e)
            n -= 1  # 如果生成失败，重试当前轮次
            continue

        acc_list = [] # 初始化准确率列表
        # 评估新解决方案，包含调试循环
        for _ in range(args.debug_max):
            try:
                # 记录评估开始时间
                eval_start_time = time.time()
                acc_list = evaluate_forward_fn(args, next_solution["code"]) # 评估新解决方案的准确率列表
                # 记录评估结束时间并计算执行时间
                eval_end_time = time.time()
                eval_execution_time = eval_end_time - eval_start_time

                
                if np.mean(acc_list) < 0.01 and SEARCHING_MODE: # 如果平均准确率过低，则抛出异常
                    raise Exception("All 0 accuracy")
                
                break # 如果评估成功，则跳出循环
            except Exception as e:
                print("During evaluation:")
                print(e)
                # 请求模型调试当前代码
                msg_list.append({"role": "assistant", "content": str(next_solution)})
                msg_list.append({"role": "user", "content": f"Error during evaluation:\n{e}\nCarefully consider where you went wrong in your latest implementation. Using insights from previous attempts, try to debug the current code to implement the same thought. Repeat your previous thought in 'thought', and put your thinking for debugging in 'debug_thought'"})
                try:
                    next_solution = get_json_response_from_gpt_reflect(msg_list, args.model)
                except Exception as e:
                    print("During LLM generate new solution:")
                    print(e)
                    continue
                continue
        
        # 如果评估失败，重试当前轮次
        if not acc_list:
            n -= 1
            continue

        fitness_str, lower_bound, upper_bound = bootstrap_confidence_interval(acc_list) # 计算准确率的置信区间
        next_solution['fitness'] = fitness_str # 设置解决方案的适应度值
        next_solution['generation'] = n + 1 # 设置解决方案的代数    
        save_new_component(next_solution, eval_execution_time, np.mean(acc_list), lower_bound, upper_bound) # 保存新的组件
        # 清理临时字段
        if 'debug_thought' in next_solution: # 如果解决方案中包含调试思路，则删除
            del next_solution['debug_thought']
        if 'reflection' in next_solution: # 如果解决方案中包含反思内容，则删除
            del next_solution['reflection']
        
        # 将新的解决方案添加到存档中
        archive.append(next_solution)

        # 保存搜索结果
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as json_file:
            json.dump(archive, json_file, indent=4)


def evaluate(args):
    """
    评估搜索结果
    
    这个函数对搜索过程中生成的所有解决方案进行完整评估，
    使用测试集计算最终的性能指标。
    
    Args:
        args: 命令行参数对象
    """
    # 构建保存搜索结果的文件路径
    file_path = os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")
    # 构建评估结果的文件路径
    eval_file_path = str(os.path.join(args.save_dir, f"{args.expr_name}_run_archive.json")).strip(".json") + "_evaluate.json"
    
    # 读取搜索结果存档
    with open(file_path, 'r') as json_file:
        archive = json.load(json_file)

    eval_archive = [] # 初始化评估结果存档
    # if os.path.exists(eval_file_path): # 如果评估结果存档已存在，则读取
    #     with open(eval_file_path, 'r') as json_file:
    #         eval_archive = json.load(json_file)

    current_idx = 0 # 遍历搜索结果存档中的解决方案
    while (current_idx < len(archive)):
        # 重新读取搜索结果存档（可能在运行过程中更新）
        with open(file_path, 'r') as json_file:
            archive = json.load(json_file)
        
        if current_idx < len(eval_archive):  # 如果当前索引小于已评估的解决方案数量，则跳过
            current_idx += 1
            continue
        
        sol = archive[current_idx] # 获取当前解决方案
        print(f"current_gen: {sol['generation']}, current_idx: {current_idx}")
        current_idx += 1
        try:
            # 记录评估开始时间
            eval_start_time = time.time()
            acc_list = evaluate_forward_fn(args, sol["code"]) # 获取当前解决方案的准确率列表
            # 记录评估结束时间并计算执行时间
            eval_end_time = time.time()
            eval_execution_time = eval_end_time - eval_start_time
            print(f"评估执行时间: {eval_execution_time:.4f} 秒")
        except Exception as e:
            print(e)
            continue
        
        fitness_str = bootstrap_confidence_interval(acc_list) # 计算准确率的置信区间
        sol['test_fitness'] = fitness_str # 设置解决方案的测试适应度值
        sol['execute_time'] = eval_execution_time
        # sol['acc_score'] = np.mean(acc_list)
        # sol['min_valid_prob'] = np.min(acc_list)
        # sol['max_valid_prob'] = np.max(acc_list)
        eval_archive.append(sol) # 将解决方案添加到评估结果存档

        # 保存评估结果
        os.makedirs(os.path.dirname(eval_file_path), exist_ok=True)
        with open(eval_file_path, 'w') as json_file:
            json.dump(eval_archive, json_file, indent=4)


def evaluate_forward_fn(args, forward_str):
    """
    评估前向函数
    
    这个函数动态执行解决方案代码，并在数据集上评估其性能。
    代码执行过程中会创建forward函数并将其绑定到AgentSystem类。
    
    Args:
        args: 命令行参数对象
        forward_str: 包含forward函数定义的代码字符串
        
    Returns:
        list: 每个样本的F1分数列表
    """
    # 动态定义forward()函数
    # 修改自 https://github.com/luchris429/DiscoPOP/blob/main/scripts/launch_evo.py
    namespace = {}
    exec(forward_str, globals(), namespace) # 执行forward_str代码，定义forward函数
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(AgentSystem, "forward", func)  # 将forward函数设置为AgentSystem类的属性

    # 设置随机种子，用于数据集划分
    examples = load_drop(args.data_filename)[1:-1] # 加载DROP数据集，排除第一个和最后一个示例（用于少样本示例）
    random.seed(args.shuffle_seed)
    random.shuffle(examples) # 打乱示例顺序

    # 根据搜索模式选择评估集合
    if SEARCHING_MODE:
        examples = examples[:args.valid_size] * args.n_repreat  # 使用验证集进行评估
    else:
        examples = examples[args.valid_size:args.valid_size + args.test_size] * args.n_repreat # 使用测试集进行评估

    questions = [example['inputs'] for example in examples]  # 提取问题
    answers = [example['targets'] for example in examples] # 提取答案

    print(f"problem length: {len(examples)}")  # 打印问题数量
    max_workers = min(len(examples), args.max_workers) if args.multiprocessing else 1 # 设置最大工作线程数

    # 构建任务队列
    task_queue = []
    for q in questions:
        taskInfo = Info('task', 'User', q, -1) # 创建任务信息
        task_queue.append(taskInfo) # 将任务信息添加到任务队列

    agentSystem = AgentSystem() # 创建AgentSystem实例

    acc_list = []  # 初始化准确率列表
    # 并行执行评估任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(agentSystem.forward, task_queue), total=len(task_queue)))

    # 处理评估结果
    for q_idx, res in enumerate(results):
        try:
            if isinstance(res, Info):
                extracted_answer = res.content # 如果结果为Info类型，则提取内容
            else:
                extracted_answer = res # 否则直接使用结果
            correct_answers = answers[q_idx] # 获取正确答案
            em_score, f1_score = drop_metric(extracted_answer, correct_answers)  # 计算准确率和F1分数
        except Exception as e:
            acc_list.append(0)  # 如果评估失败，记录为0分
            continue

        acc_list.append(f1_score)  # 将F1分数添加到准确率列表
    print(f"f1: {bootstrap_confidence_interval(acc_list)}") # 打印F1分数的置信区间
    return acc_list


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default="dataset/drop_v0_dev.jsonl.gz")  # 数据集文件路径
    parser.add_argument('--valid_size', type=int, default=128)      # 验证集大小
    parser.add_argument('--test_size', type=int, default=800)       # 测试集大小
    parser.add_argument('--shuffle_seed', type=int, default=0)      # 数据shuffle的随机种子
    parser.add_argument('--n_repreat', type=int, default=1)         # 重复评估次数
    parser.add_argument('--multiprocessing', action='store_true', default=True)  # 是否使用多进程
    parser.add_argument('--max_workers', type=int, default=48)      # 最大工作线程数
    parser.add_argument('--debug', action='store_true', default=True)  # 是否开启调试模式
    parser.add_argument('--save_dir', type=str, default='results/')     # 结果保存目录
    parser.add_argument('--expr_name', type=str, default="drop_gpt3.5_results")  # 实验名称
    parser.add_argument('--n_generation', type=int, default=10)     # 搜索代数
    parser.add_argument('--debug_max', type=int, default=3)         # 最大调试次数
    parser.add_argument('--model',                                  # 使用的模型
                        type=str,
                        default='gpt-4o-2024-05-13',
                        choices=['gpt-4-turbo-2024-04-09', 'gpt-3.5-turbo-0125', 'gpt-4o-2024-05-13'])

    args = parser.parse_args()
    
    # 执行搜索阶段
    # SEARCHING_MODE = True
    # search(args)

    # # 执行评估阶段
    SEARCHING_MODE = False
    evaluate(args)
