import json
import os
from threading import Lock
from typing import Optional, Dict, Any

class MitConfigTools:
    _cache = {
        'data': {},
        'lock': Lock()
    }

    @classmethod
    def init_config(cls, path: str, tool_name: str) -> bool:
        """初始化配置文件
        
        Args:
            path: 配置文件路径
            tool_name: 工具名称
            
        Returns:
            bool: 是否初始化成功
        """
        try:
            if not os.path.exists(path):
                return False
            
            with cls._cache['lock']:
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                result = cls._get_impl(config_data, tool_name)
                if not result:
                    return False
                
                cls._cache['data'] = result
                return True
        except Exception:
            return False

    @classmethod
    def get_param(cls, value_key: str) -> str:
        """获取配置参数
        
        Args:
            value_key: 参数键名
            
        Returns:
            str: 参数值的JSON字符串表示
        """
        with cls._cache['lock']:
            result = cls._get_impl(cls._cache['data'], value_key)
            # return json.dumps(result) if result else ""
            return result

    @classmethod
    def _get_impl(cls, src: Dict[str, Any], target: str) -> Optional[Dict[str, Any]]:
        """递归查找目标配置
        
        Args:
            src: 源配置字典
            target: 目标键名
            
        Returns:
            找到的配置值或None
        """
        if isinstance(src, dict):
            # 先在当前层级查找
            if target in src:
                return src[target]
            
            # 递归查找子层级
            for value in src.values():
                result = cls._get_impl(value, target)
                if result is not None:
                    return result
        elif isinstance(src, list):
            # 如果是列表，遍历每个元素查找
            for item in src:
                result = cls._get_impl(item, target)
                if result is not None:
                    return result
        return None


# 使用示例
if __name__ == "__main__":
    # 初始化配置
    config_path = "config.json"
    tool_name = "my_tool"
    if MitConfigTools.init_config(config_path, tool_name):
        print("配置初始化成功")
        
        # 获取参数
        param_value = MitConfigTools.get_param("some_key")
        print(f"获取到的参数值: {param_value}")
    else:
        print("配置初始化失败")