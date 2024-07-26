import os


def get_project_root():
    """获取当前项目根目录"""
    current_file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
    current_directory = os.path.dirname(current_file_path)  # 获取当前文件所在的目录
    return current_directory  # 返回项目根目录
