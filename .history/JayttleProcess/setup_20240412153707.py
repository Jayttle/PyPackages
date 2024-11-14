from setuptools import setup, find_packages

setup(
    name='JayttleProcess',
    version='0.1.6',
    description='modifty time:2024-04-06 23:00\n en: Data Process;\n zh_CN:数据处理的方法',
    packages=find_packages(),
    install_requires=[
        'pyautogui',
        'matplotlib',
        'statsmodels',
        'scikit-learn',
        'EMD-signal',
        'PyWavelets',
        'pymysql',
        'numpy',
        'seaborn',
        'pandas',
        'wordcloud',
        'tqdm',
        'paddlepaddle',
        'paddlenlp',
        # 添加其他依赖项（如果有的话）
    ],
    author='Jayttle',
    author_email='294448068@qq.com',
    license='',
)