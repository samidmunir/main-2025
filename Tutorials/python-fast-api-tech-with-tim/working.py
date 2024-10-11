"""
    Data exchange over HTTP protocol gets converted to JSON.
    - FastAPI automatically parses JSON data to internal Python types.
"""

from fastapi import FastAPI, Path
from typing import Optional
from pydantic import BaseModel

# API initialization using FastAPI.
app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    brand: Optional[str] = None

# Using a grocery store's inventory as an example.
inventory = {}

# Default endpoint for the API.
@app.get('/')
def home():
    return {'Data': 'Test'}

# Endpoint for getting the inventory.
@app.get('/get-items')
def get_items():
    return inventory

# Endpoint for getting a specific item from the inventory using its ID.
@app.get('/get-item/{item_id}')
def get_item(item_id: int):
    return inventory[item_id]

"""
@app.get('/get-item/{item_id}')
def get_item(item_id: int = Path(None, description = 'The ID of the item you would like to view.')):
    return inventory[item_id]
"""

# Endpoint for getting a specifc item from the inventory using its name.
# - If you set a query parameter = None, then it will automatically be optional.
@app.get('/get-by-name')
def get_item(name: Optional[str] = None):
    for item_id in inventory:
        if inventory[item_id].name == name:
            return inventory[item_id]
    return {'Data': 'Not found'}

# Endpoint for creating a new item in the inventory.
@app.post('/create-item/{item_id}')
def create_item(item_id: int, item: Item):
    if item_id in inventory:
        return {'Error': 'Item ID already exists.'}
    # inventory[item_id] = {'name': item.name, 'price': item.price, 'brand': item.brand}
    inventory[item_id] = item
    return inventory[item_id]