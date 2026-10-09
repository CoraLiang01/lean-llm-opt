LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"Capacity": "875"}}\n\nproducts.csv\n{"values": {"ProductName": "Spinach", "Weight": "230", "Value": "64"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "Value": "75"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "Value": "68"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "Value": "11"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "Value": "91"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "Value": "31"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "Value": "90"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "Value": "56"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "Value": "10"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "Value": "24"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '875'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
from gurobipy import Model, GRB

def solve_supermarket_optimization():
    global LEGACY_RECORDS
    capacities = []
    for rec in LEGACY_RECORDS:
        if rec.get('source') == 'capacity.csv':
            cap = rec['values'].get('Capacity')
            if cap is None:
                raise ValueError("Missing 'Capacity' in capacity.csv record")
            capacities.append(int(cap))
    if not capacities:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    total_capacity = sum(capacities)
    products = []
    for rec in LEGACY_RECORDS:
        if rec.get('source') == 'products.csv':
            v = rec['values']
            pname = v.get('ProductName')
            weight = v.get('Weight')
            value = v.get('Value')
            if pname is None or weight is None or value is None:
                raise ValueError('Missing product data in products.csv record')
            products.append({'ProductName': pname, 'Weight': int(weight), 'Value': int(value)})
    if len(products) != 10:
        raise ValueError('Expected 10 products, got %d' % len(products))
    product_keys = list(range(len(products)))
    weights = {i: products[i]['Weight'] for i in product_keys}
    values = {i: products[i]['Value'] for i in product_keys}
    names = {i: products[i]['ProductName'] for i in product_keys}
    if not len(weights) == len(values) == len(names) == 10:
        raise ValueError('Coefficient dimension mismatch')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[i] * x[i] for i in product_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[i] * x[i] for i in product_keys)) <= total_capacity, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal:', m.ObjVal)
        for i in product_keys:
            print(f'x[{i}] ({names[i]}):', x[i].X)
    else:
        print('Solver status:', m.Status)
m = solve_supermarket_optimization()