import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'Spinach', 'Weight': 282, 'Value': 49}, {'ProductName': 'Shiitake Mushrooms', 'Weight': 83, 'Value': 30}, {'ProductName': 'Apples', 'Weight': 251, 'Value': 30}, {'ProductName': 'Carrots', 'Weight': 257, 'Value': 18}, {'ProductName': 'Basil', 'Weight': 88, 'Value': 54}, {'ProductName': 'Potatoes', 'Weight': 52, 'Value': 27}, {'ProductName': 'Green Beans', 'Weight': 198, 'Value': 91}, {'ProductName': 'Blueberries', 'Weight': 203, 'Value': 88}, {'ProductName': 'Oranges', 'Weight': 87, 'Value': 78}, {'ProductName': 'Watermelons', 'Weight': 265, 'Value': 22}]
capacity = 1035
product_ids = list(range(1, 11))
product_names = {i: products[i - 1]['ProductName'] for i in product_ids}
weights = {i: products[i - 1]['Weight'] for i in product_ids}
values = {i: products[i - 1]['Value'] for i in product_ids}
if not len(product_ids) == len(weights) == len(values) == 10:
    raise ValueError('Missing product data for some indices.')

def build_model():
    m = gp.Model('Supermarket_Inventory')
    x_vars = m.addVars(product_ids, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((values[i] * x_vars[i] for i in product_ids)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weights[i] * x_vars[i] for i in product_ids)) <= capacity, name='cap')
    m.Params.MIPGap = 0.0001
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in product_ids:
        print(f"x[{i}] ({product_names[i]}): {m.getVarByName(f'x[{i}]').X}")
else:
    print(f'Solver status: {m.Status}')