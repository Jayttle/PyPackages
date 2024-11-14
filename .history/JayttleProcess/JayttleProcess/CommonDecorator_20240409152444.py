import functools
import time
from datetime import datetime


def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 记录函数调用时间
        call_time = datetime.now()
        start_time = time.time()  # 记录函数开始执行的时间

        print(f"Function '{func.__name__}' called at {call_time}")

        # 记录传入的参数
        args_str = ', '.join(map(repr, args))
        kwargs_str = ', '.join(f"{key}={value!r}" for key, value in kwargs.items())
        all_args = ', '.join(filter(None, [args_str, kwargs_str]))
        print(f"Arguments: {all_args}")

        # 调用函数并记录返回值
        result = func(*args, **kwargs)
        print(f"Returned: {result}")
        end_time = time.time()  # 记录函数执行完毕的时间
        print(f"executed in {(end_time - start_time):.4f}s")  # 打印执行时间
        return result

    return wrapper
