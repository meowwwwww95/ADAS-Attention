from typing import List, Dict, Optional

class ComponentManager:
    def __init__(self, components_file: str = "./_drop/drop_components.py"):
        """
        初始化组件管理器
        
        Args:
            components_file: 组件文件路径
        """
        self.components_file = components_file
        self.components = []
        self.load_components()
    
    def load_components(self):
        """从文件加载组件"""
        try:
            # 读取文件内容
            with open(self.components_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 执行文件内容获取 Components 变量
            namespace = {}
            exec(content, namespace)
            self.components = namespace.get('Components', [])
            print(f"✅ 成功加载 {len(self.components)} 个组件")
        except Exception as e:
            print(f"❌ 加载组件失败: {e}")
            self.components = []
    
    def save_components(self, components_file: str = "./_drop/drop_components_new.py"):
        """保存组件到文件"""
        try:
            # 构建文件内容
            content = "Components = [\n"
            for i, component in enumerate(self.components):
                content += "    {\n"
                content += f'        "thought": """{component["thought"]}""",\n'
                content += f'        "name": "{component["name"]}",\n'
                content += f'        "code": """{component["code"]}""",\n'
                content += f'        "execute_time": {component["execute_time"]},\n'
                content += f'        "acc_score": {component["acc_score"]},\n'
                content += f'        "min_valid_prob": {component["min_valid_prob"]},\n'
                content += f'        "max_valid_prob": {component["max_valid_prob"]}\n'
                content += "    }"
                if i < len(self.components) - 1:
                    content += ","
                content += "\n"
            content += "]\n"
            
            # 写入文件
            with open(self.components_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ 组件保存成功")
        except Exception as e:
            print(f"❌ 保存组件失败: {e}")
    
    def add_component(self, thought: str, name: str, code: str, execute_time, acc_score, min_valid_prob, max_valid_prob) -> bool:
        """
        添加新组件
        
        Args:
            thought: 组件思路描述
            name: 组件名称
            code: 组件代码
            execute_time: 执行时间
            acc_score: 准确率
            min_valid_prob: 95%置信区间的下界
            max_valid_prob: 95%置信区间的上界
            
        Returns:
            bool: 是否添加成功
        """
        # 检查名称是否已存在
        if self.find_component_by_name(name):
            print(f"❌ 组件名称 '{name}' 已存在")
            return False
        
        new_component = {
            "thought": thought,
            "name": name,
            "code": code,
            "execute_time": execute_time,
            "acc_score": acc_score,
            "min_valid_prob": min_valid_prob,
            "max_valid_prob": max_valid_prob,
        }
        
        self.components.append(new_component)
        print(f"✅ 成功添加组件: {name}")
        return True
    
    def update_component(self, thought: str, name: str, code: str, execute_time, acc_score, min_valid_prob, max_valid_prob) -> bool:
        """
        更新组件
        
        Args:
            thought: 组件思路描述
            name: 组件名称
            code: 组件代码
            execute_time: 执行时间
            acc_score: 准确率
            min_valid_prob: 95%置信区间的下界
            max_valid_prob: 95%置信区间的上界
            
        Returns:
            bool: 是否更新成功
        """
        # 检查名称是否已存在
        if not self.find_component_by_name(name):
            print(f"❌ 组件名称 '{name}' 不存在")  
            return False
        
        for component in self.components:
            if component["name"] == name:
                component["thought"] = thought
                component["code"] = code
                component["execute_time"] = execute_time
                component["acc_score"] = acc_score
                component["min_valid_prob"] = min_valid_prob
                component["max_valid_prob"] = max_valid_prob
                break
        
        print(f"✅ 成功更新组件: {name}")
        return True
    
    def find_component_by_name(self, name: str) -> Optional[Dict]:
        """
        根据名称查找组件
        
        Args:
            name: 组件名称
            
        Returns:
            Dict or None: 找到的组件或None
        """
        for component in self.components:
            if component["name"] == name:
                return component
        return None
    
    def list_components(self):
        """列出所有组件"""
        if not self.components:
            print("📝 暂无组件")
            return
        
        print(f"📋 共有 {len(self.components)} 个组件:")
        print("-" * 60)
        res = []
        for i, component in enumerate(self.components, 1):
            print(f"{i}.name: {component['name']}")
            print()
            res.append(component)
        return res

    def show_component(self, name: str):
        """显示组件详细信息"""
        component = self.find_component_by_name(name)
        if not component:
            print(f"❌ 未找到组件: {name}")
            return
        
        print("=" * 60)
        print(f"组件详情:")
        print(f"    name: {component['name']}")
        print(f"    thought: {component['thought']}")
        print(f"    code: {component['code']}")
        print(f"    execute_time: {component['execute_time']}")
        print(f"    acc_score: {component['acc_score']}")
        return component

Component_Manager = ComponentManager()
# print(Component_Manager.show_component("Chain-of-Thought"))