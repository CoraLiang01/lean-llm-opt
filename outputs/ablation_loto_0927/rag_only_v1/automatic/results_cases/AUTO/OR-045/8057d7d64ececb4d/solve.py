LEGACY_OBSERVATION = 'capacity.csv\n{"values": {"Capacity": "1035"}}\n\nproducts.csv\n{"values": {"ProductName": "Spinach", "Weight": "282", "Value": "49"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "83", "Value": "30"}}\n{"values": {"ProductName": "Apples", "Weight": "251", "Value": "30"}}\n{"values": {"ProductName": "Carrots", "Weight": "257", "Value": "18"}}\n{"values": {"ProductName": "Basil", "Weight": "88", "Value": "54"}}\n{"values": {"ProductName": "Potatoes", "Weight": "52", "Value": "27"}}\n{"values": {"ProductName": "Green Beans", "Weight": "198", "Value": "91"}}\n{"values": {"ProductName": "Blueberries", "Weight": "203", "Value": "88"}}\n{"values": {"ProductName": "Oranges", "Weight": "87", "Value": "78"}}\n{"values": {"ProductName": "Watermelons", "Weight": "265", "Value": "22"}}'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '1035'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
from gurobipy import Model, GRB

def solve_inventory_optimization():
    global LEGACY_RECORDS
    capacities = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            for (k, v) in rec['values'].items():
                capacities[k] = int(v)
    if 'Capacity' not in capacities:
        raise ValueError('Missing total capacity in LEGACY_RECORDS.')
    products = []
    weights = {}
    values = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            pname = rec['values']['ProductName']
            products.append(pname)
            try:
                weights[pname] = int(rec['values']['Weight'])
                values[pname] = int(rec['values']['Value'])
            except Exception as e:
                raise ValueError(f'Invalid data for product {pname}: {e}')
    if len(products) != 10:
        raise ValueError('Expected 10 products, got %d' % len(products))
    for pname in products:
        if pname not in weights or pname not in values:
            raise ValueError(f'Missing weight or value for product {pname}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
    m.addConstr(sum((weights[p] * x[p] for p in products)) <= capacities['Capacity'], name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for p in products:
            print(x[p].VarName, x[p].X)
    else:
        print('Solver status:', m.Status)
    return m
m = solve_inventory_optimization()