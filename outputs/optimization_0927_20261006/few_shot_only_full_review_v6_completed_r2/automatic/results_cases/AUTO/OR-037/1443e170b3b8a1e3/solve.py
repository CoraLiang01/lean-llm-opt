import gurobipy as gp
from gurobipy import GRB
vehicle_types = ['Sedan', 'SUV', 'Truck', 'Convertible', 'Minivan', 'Coupe', 'Hatchback', 'Station Wagon', 'Electric Car', 'Hybrid Car', 'Luxury Sedan', 'Sports Car', 'Crossover', 'Diesel Truck', 'Compact SUV', 'Luxury SUV', 'Cargo Van', 'Pickup Truck', 'Roadster', 'Muscle Car', 'Off-road Vehicle', 'Camper Van', 'Compact Car', 'Motorcycle', 'Electric SUV']
profit = {'Sedan': 2524, 'SUV': 4614, 'Truck': 8416, 'Convertible': 5917, 'Minivan': 9048, 'Coupe': 1140, 'Hatchback': 8962, 'Station Wagon': 1888, 'Electric Car': 8487, 'Hybrid Car': 4425, 'Luxury Sedan': 4717, 'Sports Car': 4210, 'Crossover': 1226, 'Diesel Truck': 7400, 'Compact SUV': 4639, 'Luxury SUV': 7712, 'Cargo Van': 3299, 'Pickup Truck': 9895, 'Roadster': 4496, 'Muscle Car': 4526, 'Off-road Vehicle': 5688, 'Camper Van': 3007, 'Compact Car': 3623, 'Motorcycle': 8474, 'Electric SUV': 8372}
weight = {'Sedan': 99, 'SUV': 55, 'Truck': 75, 'Convertible': 94, 'Minivan': 80, 'Coupe': 82, 'Hatchback': 71, 'Station Wagon': 100, 'Electric Car': 28, 'Hybrid Car': 93, 'Luxury Sedan': 84, 'Sports Car': 83, 'Crossover': 62, 'Diesel Truck': 90, 'Compact SUV': 99, 'Luxury SUV': 96, 'Cargo Van': 21, 'Pickup Truck': 39, 'Roadster': 99, 'Muscle Car': 81, 'Off-road Vehicle': 6, 'Camper Van': 58, 'Compact Car': 37, 'Motorcycle': 15, 'Electric SUV': 37}
capacity = 765
for v in vehicle_types:
    if v not in profit or v not in weight:
        raise ValueError(f'Missing profit or weight for vehicle type: {v}')
m = gp.Model('Car_Inventory_Optimization')
x_vars = m.addVars(vehicle_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[v] * x_vars[v] for v in vehicle_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[v] * x_vars[v] for v in vehicle_types)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in vehicle_types:
        print(f'{x_vars[v].VarName}: {x_vars[v].X}')
else:
    print(f'Solver status: {m.Status}')