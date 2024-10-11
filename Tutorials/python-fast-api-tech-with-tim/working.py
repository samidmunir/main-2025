"""
    Data exchange over HTTP protocol gets converted to JSON.
    - FastAPI automatically parses JSON data to internal Python types.
"""

from fastapi import FastAPI, Path

# API initialization using FastAPI.
app = FastAPI()

# Using a grocery store's inventory as an example.
inventory = {
    1: {
        'name': 'Milk',
        'price': 3.99,
        'brand': 'Kirkland'
    },
    2: {
        'name': 'Bread',
        'price': 2.49,
        'brand': 'Hearty Buns'
    }
}

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

# Query parameters.
@app.get('/get-by-name')
def get_item(name: str):
    for item_id in inventory:
        if inventory[item_id]['name'] == name:
            return inventory[item_id]
    return {'Data': 'Not found'}