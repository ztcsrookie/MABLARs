from setuptools import setup, find_packages

setup(
    name='mablars',
    version='0.1.7',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'causal-learn',
        'scikit-learn',
        'python-weka-wrapper3'
    ],

    description='This is a python package of mablars',
    long_description=open('README.md').read(),  # 从 README 文件读取详细描述
    long_description_content_type='text/markdown',  # README 文件的格式
    author='Te Zhang',
    author_email='ztcsrookie@gmail.com',
    url='https://github.com/ztcsrookie/mablars',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
    ],
)
