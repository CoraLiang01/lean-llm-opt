LEGACY_OBSERVATION = '{"values": {"Capacity": "875"}}\n{"values": {"ProductName": "Spinach", "Weight": "230", "Value": "64"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "Value": "75"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "Value": "68"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "Value": "11"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "Value": "91"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "Value": "31"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "Value": "90"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "Value": "56"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "Value": "10"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "Value": "24"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
from gurobipy import Model, GRB

def solve_supermarket_optimization():
    global LEGACY_RECORDS
    products = []
    capacity = None
    for rec in LEGACY_RECORDS:
        vals = rec['values']
        if 'Capacity' in vals and vals['Capacity']:
            if capacity is not None:
                raise ValueError('Multiple capacities found in LEGACY_RECORDS')
            capacity = int(vals['Capacity'])
        elif 'ProductName' in vals and vals['ProductName']:
            try:
                pname = vals['ProductName']
                weight = int(vals['Weight'])
                value = int(vals['Value'])
            except Exception as e:
                raise ValueError(f'Invalid product record: {vals}') from e
            products.append({'ProductName': pname, 'Weight': weight, 'Value': value})
    if capacity is None:
        raise ValueError('No capacity found in LEGACY_RECORDS')
    if len(products) == 0:
        raise ValueError('No products found in LEGACY_RECORDS')
    for prod in products:
        if not all((k in prod for k in ('ProductName', 'Weight', 'Value'))):
            raise ValueError(f'Missing data in product: {prod}')
    prod_keys = [prod['ProductName'] for prod in products]
    weight = {prod['ProductName']: prod['Weight'] for prod in products}
    value = {prod['ProductName']: prod['Value'] for prod in products}
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(prod_keys, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(sum((value[p] * x[p] for p in prod_keys)), GRB.MAXIMIZE)
    m.addConstr(sum((weight[p] * x[p] for p in prod_keys)) <= capacity, name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for p in prod_keys:
            print(f'{x[p].VarName}: {x[p].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_supermarket_optimization()