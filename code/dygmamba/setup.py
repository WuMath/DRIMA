from setuptools import setup, find_packages

setup(
    name="my_project",
    version="0.1",
    packages=find_packages(), # 会自动发现 src 目录下的包
)

# pip install -e .