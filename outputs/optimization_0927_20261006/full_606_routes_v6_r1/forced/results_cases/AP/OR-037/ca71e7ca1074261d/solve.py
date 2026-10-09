import gurobipy as gp
from gurobipy import GRB
products = [{'ProductName': 'Sedan', 'Value': 2524, 'Weight': 99}, {'ProductName': 'SUV', 'Value': 4614, 'Weight': 55}, {'ProductName': 'Truck', 'Value': 8416, 'Weight': 75}, {'ProductName': 'Convertible', 'Value': 5917, 'Weight': 94}, {'ProductName': 'Minivan', 'Value': 9048, 'Weight': 80}, {'ProductName': 'Coupe', 'Value': 1140, 'Weight': 82}, {'ProductName': 'Hatchback', 'Value': 8962, 'Weight': 71}, {'ProductName': 'Station Wagon', 'Value': 1888, 'Weight': 100}, {'ProductName': 'Electric Car', 'Value': 8487, 'Weight': 28}, {'ProductName': 'Hybrid Car', 'Value': 4425, 'Weight': 93}, {'ProductName': 'Luxury Sedan', 'Value': 4717, 'Weight': 84}, {'ProductName': 'Sports Car', 'Value': 4210, 'Weight': 83}, {'ProductName': 'Crossover', 'Value': 1226, 'Weight': 62}, {'ProductName': 'Diesel Truck', 'Value': 7400, 'Weight': 90}, {'ProductName': 'Compact SUV', 'Value': 4639, 'Weight': 99}, {'ProductName': 'Luxury SUV', 'Value': 7712, 'Weight': 96}, {'ProductName': 'Cargo Van', 'Value': 3299, 'Weight': 21}, {'ProductName': 'Pickup Truck', 'Value': 9895, 'Weight': 39}, {'ProductName': 'Roadster', 'Value': 4496, 'Weight': 99}, {'ProductName': 'Muscle Car', 'Value': 4526, 'Weight': 81}, {'ProductName': 'Off-road Vehicle', 'Value': 5688, 'Weight': 6}, {'ProductName': 'Camper Van', 'Value': 3007, 'Weight': 58}, {'ProductName': 'Compact Car', 'Value': 3623, 'Weight': 37}, {'ProductName': 'Motorcycle', 'Value': 8474, 'Weight': 15}, {'ProductName': 'Electric SUV', 'Value': 8372, 'Weight': 37}]
capacity = 765
product_names = [p['ProductName'] for p in products]
values = {p['ProductName']: p['Value'] for p in products}
weights = {p['ProductName']: p['Weight'] for p in products}
if not len(product_names) == len(values) == len(weights):
    raise ValueError('Mismatch in product data dimensions.')
m = gp.Model('inventory_replenishment')
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[name] * x_vars[name] for name in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[name] * x_vars[name] for name in product_names)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')