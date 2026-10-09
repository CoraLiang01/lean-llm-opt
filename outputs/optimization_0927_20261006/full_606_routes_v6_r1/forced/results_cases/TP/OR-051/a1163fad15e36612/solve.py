import gurobipy as gp
from gurobipy import GRB
cabinets = [{'CabinetID': '1', 'Capacity': 400}, {'CabinetID': '2', 'Capacity': 600}, {'CabinetID': '3', 'Capacity': 500}, {'CabinetID': '4', 'Capacity': 700}, {'CabinetID': '5', 'Capacity': 450}, {'CabinetID': '6', 'Capacity': 650}, {'CabinetID': '7', 'Capacity': 550}, {'CabinetID': '8', 'Capacity': 750}, {'CabinetID': '9', 'Capacity': 480}, {'CabinetID': '10', 'Capacity': 520}]
products = [{'ProductName': 'Espresso Beans', 'Value': 100, 'Weight': 1.0}, {'ProductName': 'Colombian Roast', 'Value': 150, 'Weight': 1.5}, {'ProductName': 'Arabica Blend', 'Value': 80, 'Weight': 1.2}, {'ProductName': 'French Roast', 'Value': 120, 'Weight': 1.3}, {'ProductName': 'Italian Roast', 'Value': 130, 'Weight': 1.4}, {'ProductName': 'House Blend', 'Value': 110, 'Weight': 1.1}, {'ProductName': 'Sumatra Coffee', 'Value': 160, 'Weight': 1.8}, {'ProductName': 'Mocha Java', 'Value': 90, 'Weight': 1.2}, {'ProductName': 'Hazelnut Flavor', 'Value': 95, 'Weight': 1.0}, {'ProductName': 'Caramel Blend', 'Value': 105, 'Weight': 1.3}, {'ProductName': 'Vanilla Flavor', 'Value': 85, 'Weight': 1.2}, {'ProductName': 'Cappuccino Mix', 'Value': 140, 'Weight': 1.5}, {'ProductName': 'Pumpkin Spice', 'Value': 75, 'Weight': 1.1}, {'ProductName': 'Decaf Roast', 'Value': 60, 'Weight': 1.0}, {'ProductName': 'Organic Roast', 'Value': 170, 'Weight': 1.6}, {'ProductName': 'Cold Brew', 'Value': 115, 'Weight': 1.4}, {'ProductName': 'Peruvian Blend', 'Value': 155, 'Weight': 1.7}, {'ProductName': 'Kenyan AA', 'Value': 125, 'Weight': 1.3}]
cabinet_ids = [c['CabinetID'] for c in cabinets]
cabinet_caps = {c['CabinetID']: float(c['Capacity']) for c in cabinets}
product_names = [p['ProductName'] for p in products]
product_values = {p['ProductName']: float(p['Value']) for p in products}
product_weights = {p['ProductName']: float(p['Weight']) for p in products}
if set(cabinet_ids) != set((str(i) for i in range(1, 11))):
    raise ValueError('Cabinet IDs do not match required set 1-10')
if len(product_names) != 18:
    raise ValueError('There must be exactly 18 products')
for p in product_names:
    if p not in product_values or p not in product_weights:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinet_ids, product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in cabinet_ids for j in product_names)), GRB.MAXIMIZE)
for i in cabinet_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_names)) <= cabinet_caps[i], name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')