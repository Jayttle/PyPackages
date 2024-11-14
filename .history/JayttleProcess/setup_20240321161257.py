from setuptools import setup, find_packages

setup(
    name='JayttleProcess',
    version='0.0.5',
    description='modifty time:2024-03-20 21:57\n en: Data Process;\n zh_CN:数据处理的方法',
    packages=find_packages(),
    install_requires=[
        'pyautogui',
        'matplotlib',
        'statsmodels',
        'scikit-learn',
        'EMD-signal',
        'PyWavelets',
        # 添加其他依赖项（如果有的话）
    ],
    author='Jayttle',
    author_email='294448068@qq.com',
    license='',
)