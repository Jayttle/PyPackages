import json
from datetime import datetime

class Product:
    def __init__(self, category: str, model: str, time: str, price: float, warranty: str, remarks: str):
        """
        初始化产品实例。
        
        参数:
            category (str): 产品类别。
            model (str): 产品型号。
            time (str): 产品生产时间，格式为 "YYYY-MM-DD"。
            price (float): 产品价格。
            warranty (str): 产品保修信息。
            remarks (str): 产品备注。
        """
        self.category = category
        self.model = model
        self.time = datetime.strptime(time, "%Y-%m-%d")
        self.price = price
        self.warranty = warranty
        self.remarks = remarks

    def __str__(self) -> str:
        """
        返回产品的字符串表示形式。
        
        返回:
            str: 产品的字符串表示形式。
        """
        return f"{self.category}\t{self.model}\t{self.time.strftime('%Y-%m-%d')}\t{self.price}\t{self.warranty}\t{self.remarks}"
    
    def to_dict(self) -> dict:
        """
        将产品实例转换为字典。
        
        返回:
            dict: 包含产品信息的字典。
        """
        return {
            "category": self.category,
            "model": self.model,
            "time": self.time.strftime('%Y-%m-%d'),
            "price": self.price,
            "warranty": self.warranty,
            "remarks": self.remarks
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Product':
        """
        从字典创建产品实例。
        
        参数:
            data (dict): 包含产品信息的字典。
        
        返回:
            Product: 产品实例。
        """
        return cls(
            category=data["category"],
            model=data["model"],
            time=data["time"],
            price=data["price"],
            warranty=data["warranty"],
            remarks=data["remarks"]
        )
    
    
    def days_of_usage(self) -> int:
        """
        计算从生产时间到今天的使用天数。
        
        返回:
            int: 使用天数。
        """
        today = datetime.now()
        days = (today - self.time).days
        return days if days > 0 else 0

    def daily_price(self) -> float:
        """
        计算每日平均价格。
        
        返回:
            float: 每日平均价格。
        """
        today = datetime.now()
        days = (today - self.time).days
        if days == 0:
            return self.price
        else:
            return round(self.price / days, 2)

class ProductList:
    def __init__(self):
        """
        初始化产品列表。
        """
        self.products: list[Product] = []

    def add_product(self, product: Product) -> None:
        """
        向产品列表中添加产品。
        
        参数:
            product (Product): 要添加的产品。
        """
        self.products.append(product)

    def read_data(self, file_path: str) -> None:
        """
        从文件中读取数据，并将其转换为产品列表。
        
        参数:
            file_path (str): 包含数据的文件路径。
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = line.strip().split('\t')
                category = data[0]
                model = data[1]
                time = data[2]
                price = float(data[3])
                warranty = data[4]
                remarks = data[5]
                self.add_product(Product(category, model, time, price, warranty, remarks))

    def save_data(self, file_path: str) -> None:
        """
        将产品列表保存为文本文件。
        
        参数:
            file_path (str): 要保存的文件路径。
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            for product in self.products:
                file.write(str(product) + '\n')

    def save_data(self, file_path: str) -> None:
        """
        将产品列表保存为 JSON 文件。
        
        参数:
            file_path (str): 要保存的文件路径。
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([product.to_dict() for product in self.products], file, ensure_ascii=False, indent=4)

    def load_from_json(self, file_path: str) -> None:
        """
        从 JSON 文件加载数据，并将其转换为产品列表。
        
        参数:
            file_path (str): 包含数据的文件路径。
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.products = [Product.from_dict(item) for item in data]

    def calculate_usage_and_price(self) -> None:
        """
        计算每个设备的使用天数和日均价格，并打印结果。
        """
        for product in self.products:
            days_usage = product.days_of_usage()
            daily_price = product.daily_price()
            print(f"{product.category}\t{product.model}\t{product.time.strftime('%Y-%m-%d')}\t价格：{product.price} 元\t使用天数：{days_usage} 天\t日均价格：{daily_price} 元")


product_list = ProductList()
file_path = "D:/Program Files (x86)/Software/OneDrive/设备.txt"
product_list.read_data(file_path)
product_list.calculate_usage_and_price()
