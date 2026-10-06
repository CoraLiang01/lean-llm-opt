LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"Capacity": "180"}}\n\nproducts.csv\n{"values": {"ProductName": "Baguette", "Value": "888", "Weight": "4"}}\n{"values": {"ProductName": "Croissant", "Value": "134", "Weight": "2"}}\n{"values": {"ProductName": "Sourdough", "Value": "129", "Weight": "4"}}\n{"values": {"ProductName": "Rye Bread", "Value": "370", "Weight": "3"}}\n{"values": {"ProductName": "Brioche", "Value": "921", "Weight": "2"}}\n{"values": {"ProductName": "Focaccia", "Value": "765", "Weight": "1"}}\n{"values": {"ProductName": "Ciabatta", "Value": "154", "Weight": "2"}}\n{"values": {"ProductName": "Pita", "Value": "837", "Weight": "1"}}\n{"values": {"ProductName": "Bagel", "Value": "584", "Weight": "3"}}\n{"values": {"ProductName": "English Muffin", "Value": "365", "Weight": "3"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '180'}}, {'source': 'products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
from gurobipy import Model, GRB

def solve_bakery_optimization():
    records = LEGACY_RECORDS
    capacity_records = [r for r in records if r['source'] == 'capacity.csv']
    if not capacity_records or 'Capacity' not in capacity_records[0]['values']:
        raise ValueError('Missing capacity data')
    total_capacity = int(capacity_records[0]['values']['Capacity'])
    product_records = [r for r in records if r['source'] == 'products.csv']
    products = []
    for r in product_records:
        vals = r['values']
        if not all((k in vals for k in ('ProductName', 'Value', 'Weight'))):
            raise ValueError(f'Missing data in product record: {vals}')
        products.append({'ProductName': vals['ProductName'], 'Value': int(vals['Value']), 'Weight': int(vals['Weight'])})
    required_names = ['Baguette', 'Croissant', 'Sourdough', 'Rye Bread', 'Brioche', 'Focaccia', 'Ciabatta', 'Pita', 'Bagel', 'English Muffin']
    product_names = [p['ProductName'] for p in products]
    missing = set(required_names) - set(product_names)
    if missing:
        raise ValueError(f'Missing products: {missing}')
    business_keys = {}
    for p in products:
        key = p['ProductName'].replace(' ', '')
        business_keys[p['ProductName']] = key
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = {}
    for p in products:
        key = business_keys[p['ProductName']]
        x[key] = m.addVar(vtype=GRB.INTEGER, lb=0, name=key)
    obj = 0
    for p in products:
        key = business_keys[p['ProductName']]
        obj += p['Value'] * x[key]
    m.setObjective(obj, GRB.MAXIMIZE)
    expr = 0
    for p in products:
        key = business_keys[p['ProductName']]
        expr += p['Weight'] * x[key]
    m.addConstr(expr <= total_capacity, name='storage')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for p in products:
            key = business_keys[p['ProductName']]
            var = x[key]
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_bakery_optimization()