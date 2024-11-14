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