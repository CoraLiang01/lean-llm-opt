import gurobipy as gp
from gurobipy import GRB
products = [{'Product Name': 'Baby Food_255.28', 'Revenue': 255.28, 'Demand': 3066513, 'Initial Inventory': 22749210}]
for prod in products:
    if not all((k in prod for k in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'])):
        raise ValueError(f'Missing data for product: {prod}')
product_names = [prod['Product Name'] for prod in products]
revenue = {prod['Product Name']: prod['Revenue'] for prod in products}
demand = {prod['Product Name']: prod['Demand'] for prod in products}
initial_inventory = {prod['Product Name']: prod['Initial Inventory'] for prod in products}
m = gp.Model('Baby_Product_Revenue_Maximization')
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, ub={name: min(demand[name], initial_inventory[name]) for name in product_names}, name='')
m.setObjective(gp.quicksum((revenue[name] * x_vars[name] for name in product_names)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for name in product_names:
        print(f'{x_vars[name].VarName}: {x_vars[name].X}')
else:
    print(f'Solver status: {m.Status}')