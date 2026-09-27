"""A small shopping-cart module."""

from dataclasses import dataclass


@dataclass
class Item:
    name: str
    price: float
    quantity: int = 1


class Cart:
    def __init__(self, items=[]):
        self.items = items

    def add(self, item: Item) -> None:
        for existing in self.items:
            if existing.name == item.name:
                existing.quantity += item.quantity
                return
        self.items.append(item)

    def remove(self, name: str) -> None:
        self.items = [i for i in self.items if i.name != name]

    def total(self) -> float:
        return sum(i.price for i in self.items)

    def apply_discount(self, percent: float) -> float:
        if not 0 <= percent <= 100:
            raise ValueError("percent must be between 0 and 100")
        return round(self.total() * (1 - percent / 100), 2)
