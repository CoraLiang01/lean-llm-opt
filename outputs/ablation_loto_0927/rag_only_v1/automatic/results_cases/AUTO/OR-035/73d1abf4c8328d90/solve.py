LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"Capacity": "180"}}\n\nproducts.csv\n{"values": {"ProductName": "Baguette", "Value": "888", "Weight": "4"}}\n{"values": {"ProductName": "Croissant", "Value": "134", "Weight": "2"}}\n{"values": {"ProductName": "Sourdough", "Value": "129", "Weight": "4"}}\n{"values": {"ProductName": "Rye Bread", "Value": "370", "Weight": "3"}}\n{"values": {"ProductName": "Brioche", "Value": "921", "Weight": "2"}}\n{"values": {"ProductName": "Focaccia", "Value": "765", "Weight": "1"}}\n{"values": {"ProductName": "Ciabatta", "Value": "154", "Weight": "2"}}\n{"values": {"ProductName": "Pita", "Value": "837", "Weight": "1"}}\n{"values": {"ProductName": "Bagel", "Value": "584", "Weight": "3"}}\n{"values": {"ProductName": "English Muffin", "Value": "365", "Weight": "3"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '180'}}, {'source': 'products.csv', 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}]
from gurobipy import Model, GRB

def solve_bakery_optimization():
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            v = rec['values']
            products.append({'ProductName': v['ProductName'], 'Value': int(v['Value']), 'Weight': int(v['Weight'])})
        elif rec['source'] == 'capacity.csv':
            if capacity is not None:
                raise ValueError('Multiple capacity records found')
            capacity = int(rec['values']['Capacity'])
    if capacity is None:
        raise ValueError('No capacity record found')
    if len(products) == 0:
        raise ValueError('No products found')
    for p in products:
        if not all((k in p for k in ('ProductName', 'Value', 'Weight'))):
            raise ValueError(f'Missing fields in product: {p}')
    product_names = [p['ProductName'] for p in products]
    profit = {p['ProductName']: p['Value'] for p in products}
    weight = {p['ProductName']: p['Weight'] for p in products}
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((profit[name] * x[name] for name in product_names)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[name] * x[name] for name in product_names)) <= capacity, name='storage')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for name in product_names:
            print(f'{x[name].VarName} {x[name].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_bakery_optimization()