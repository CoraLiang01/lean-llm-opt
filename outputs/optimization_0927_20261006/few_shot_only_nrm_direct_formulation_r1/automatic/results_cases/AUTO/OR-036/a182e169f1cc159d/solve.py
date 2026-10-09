import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedan', 'SUV', 'Truck', 'Convertible', 'Minivan', 'Coupe', 'Hatchback', 'Station Wagon', 'Electric Car', 'Hybrid Car', 'Luxury Sedan', 'Sports Car', 'Crossover', 'Diesel Truck', 'Compact SUV', 'Luxury SUV', 'Cargo Van', 'Pickup Truck', 'Roadster', 'Muscle Car', 'Off-road Vehicle', 'Camper Van', 'Compact Car', 'Motorcycle', 'Electric SUV']
values = {'Sedan': 1752, 'SUV': 1856, 'Truck': 8372, 'Convertible': 6168, 'Minivan': 9681, 'Coupe': 8062, 'Hatchback': 3895, 'Station Wagon': 3254, 'Electric Car': 1701, 'Hybrid Car': 6799, 'Luxury Sedan': 2724, 'Sports Car': 6304, 'Crossover': 3255, 'Diesel Truck': 1923, 'Compact SUV': 4103, 'Luxury SUV': 4429, 'Cargo Van': 2663, 'Pickup Truck': 1691, 'Roadster': 5632, 'Muscle Car': 4793, 'Off-road Vehicle': 1343, 'Camper Van': 9124, 'Compact Car': 3652, 'Motorcycle': 8842, 'Electric SUV': 9176}
weights = {'Sedan': 15, 'SUV': 87, 'Truck': 36, 'Convertible': 30, 'Minivan': 33, 'Coupe': 72, 'Hatchback': 75, 'Station Wagon': 71, 'Electric Car': 51, 'Hybrid Car': 21, 'Luxury Sedan': 97, 'Sports Car': 52, 'Crossover': 25, 'Diesel Truck': 15, 'Compact SUV': 54, 'Luxury SUV': 57, 'Cargo Van': 18, 'Pickup Truck': 69, 'Roadster': 26, 'Muscle Car': 38, 'Off-road Vehicle': 31, 'Camper Van': 74, 'Compact Car': 82, 'Motorcycle': 49, 'Electric SUV': 64}
capacity = 1576
if set(values.keys()) != set(vehicle_types):
    raise ValueError('Mismatch between vehicle_types and values keys')
if set(weights.keys()) != set(vehicle_types):
    raise ValueError('Mismatch between vehicle_types and weights keys')
m = gp.Model('vehicle_inventory_optimization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x_vars[i] for i in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x_vars[i] for i in vehicle_types)) <= capacity, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')