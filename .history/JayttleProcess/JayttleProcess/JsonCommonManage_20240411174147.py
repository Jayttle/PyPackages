import json
from datetime import datetime

class Product:
    def __init__(self, category: str, model: str, time: str, price: float, warranty: str, remarks: str):
        self.category = category
        self.model = model
        self.time = time
        self.price = price
        self.warranty = warranty
        self.remarks = remarks

    def __str__(self) -> str:
        return f"{self.category}-{self.model}-{self.time}-{self.price}-{self.warranty}-{self.remarks}"
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "model": self.model,
            "time": self.time,
            "price": self.price,
            "warranty": self.warranty,
            "remarks": self.remarks
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Product':
        return cls(
            category=data["category"],
            model=data["model"],
            time=data["time"],
            price=data["price"],
            warranty=data["warranty"],
            remarks=data["remarks"]
        )

class ProductList:
    def __init__(self):
        self.products: list[Product] = []

    def add_product(self, product: Product) -> None:
        self.products.append(product)

    def read_data(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = line.strip().split('-')
                category = data[0]
                model = data[1]
                time = data[2]
                price = float(data[3])
                warranty = data[4]
                remarks = data[5]
                self.add_product(Product(category, model, time, price, warranty, remarks))

    def save_data(self, file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as file:
            for product in self.products:
                file.write(str(product) + '\n')

    def save_data(self, file_path: str) -> None:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([product.to_dict() for product in self.products], file, ensure_ascii=False, indent=4)

    def load_from_json(self, file_path: str) -> None:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.products = [Product.from_dict(item) for item in data]


product_list = ProductList()
file_path = "D:/Program Files (x86)/Software/OneDrive/设备.txt"
product_list.read_data(file_path)
file_path_json = "D:/Program Files (x86)/Software/OneDrive/设备清单.json"
product_list.save_data(file_path_json)
