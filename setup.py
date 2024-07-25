from setuptools import setup, find_packages

setup(
    name='mablars',  # 库的名称
    version='0.1.0',  # 库的版本
    packages=find_packages(),  # 自动发现所有的包
    install_requires=[
        # 在这里列出你的库依赖的第三方包，例如:
        # 'requests',
    ],
    description='This is a python package of mablars',  # 库的简短描述
    long_description=open('README.md').read(),  # 从 README 文件读取详细描述
    long_description_content_type='text/markdown',  # README 文件的格式
    author='Te Zhang',  # 你的名字
    author_email='ztcsrookie@gmail.com',  # 你的邮箱
    url='https://github.com/yourusername/my_library',  # 你的库的主页
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
