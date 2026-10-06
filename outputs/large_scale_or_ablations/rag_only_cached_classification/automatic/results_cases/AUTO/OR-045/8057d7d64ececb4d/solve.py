LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"Capacity": "1035"}}\n\nproducts.csv\n{"values": {"ProductName": "Spinach", "Weight": "282", "Value": "49"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "83", "Value": "30"}}\n{"values": {"ProductName": "Apples", "Weight": "251", "Value": "30"}}\n{"values": {"ProductName": "Carrots", "Weight": "257", "Value": "18"}}\n{"values": {"ProductName": "Basil", "Weight": "88", "Value": "54"}}\n{"values": {"ProductName": "Potatoes", "Weight": "52", "Value": "27"}}\n{"values": {"ProductName": "Green Beans", "Weight": "198", "Value": "91"}}\n{"values": {"ProductName": "Blueberries", "Weight": "203", "Value": "88"}}\n{"values": {"ProductName": "Oranges", "Weight": "87", "Value": "78"}}\n{"values": {"ProductName": "Watermelons", "Weight": "265", "Value": "22"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '1035'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
from gurobipy import Model, GRB

def solve_inventory_optimization():
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            capacity = int(rec['values']['Capacity'])
        elif rec['source'] == 'products.csv':
            products.append({'ProductName': rec['values']['ProductName'], 'Weight': int(rec['values']['Weight']), 'Value': int(rec['values']['Value'])})
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) == 0:
        raise ValueError('No products found in LEGACY_RECORDS')
    for p in products:
        if not all((k in p for k in ('ProductName', 'Weight', 'Value'))):
            raise ValueError(f'Missing data in product: {p}')
    prod_keys = [p['ProductName'] for p in products]
    weights = {p['ProductName']: p['Weight'] for p in products}
    values = {p['ProductName']: p['Value'] for p in products}
    if set(weights.keys()) != set(prod_keys) or set(values.keys()) != set(prod_keys):
        raise ValueError('Coefficient keys do not match product keys')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(prod_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((values[k] * x[k] for k in prod_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[k] * x[k] for k in prod_keys)) <= capacity, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for k in prod_keys:
            print(f'{x[k].VarName} {x[k].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_inventory_optimization()