LEGACY_OBSERVATION = 'capacity.csv\nCapacity\n875\n\nproducts.csv\nProductName,Weight,Value\nSpinach,230,64\nShiitake Mushrooms,637,75\nApples,773,68\nCarrots,653,11\nBasil,755,91\nPotatoes,670,31\nGreen Beans,505,90\nBlueberries,821,56\nOranges,83,10\nWatermelons,249,24'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'Capacity': '875'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
from gurobipy import Model, GRB

def solve_supermarket_optimization():
    global LEGACY_RECORDS
    products = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            pname = rec['values']['ProductName']
            try:
                weight = int(rec['values']['Weight'])
                value = int(rec['values']['Value'])
            except Exception:
                raise ValueError(f'Non-integer weight or value for product {pname}')
            products.append({'ProductName': pname, 'Weight': weight, 'Value': value})
    if len(products) != 10:
        raise ValueError(f'Expected 10 products, got {len(products)}')
    capacities = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            try:
                cap = int(rec['values']['Capacity'])
            except Exception:
                raise ValueError('Non-integer capacity in capacity.csv')
            capacities.append(cap)
    if len(capacities) != 1:
        raise ValueError(f'Expected 1 capacity, got {len(capacities)}')
    total_capacity = capacities[0]
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(range(len(products)), vtype=GRB.INTEGER, lb=0, name='')
    obj = sum((products[i]['Value'] * x[i] for i in range(len(products))))
    m.setObjective(obj, GRB.MAXIMIZE)
    total_weight = sum((products[i]['Weight'] * x[i] for i in range(len(products))))
    m.addConstr(total_weight <= total_capacity, name='stock_capacity')
    if any(('Weight' not in prod or 'Value' not in prod for prod in products)):
        raise ValueError('Missing Weight or Value in product data')
    if total_capacity is None:
        raise ValueError('Missing total capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i, prod in enumerate(products):
            var = x[i]
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_supermarket_optimization()