import gurobipy as gp
from gurobipy import GRB
I = [1, 2, 3, 4, 5, 6]
revenue = {1: 119.144, 2: 120.144, 3: 121.244, 4: 119.144, 5: 120.144, 6: 121.244}
inventory = {1: 200, 2: 150, 3: 150, 4: 200, 5: 150, 6: 150}
demand = {1: 30, 2: 50, 3: 10, 4: 30, 5: 50, 6: 10}
if set(revenue.keys()) != set(I):
    raise ValueError('Revenue data missing for some indices.')
if set(inventory.keys()) != set(I):
    raise ValueError('Inventory data missing for some indices.')
if set(demand.keys()) != set(I):
    raise ValueError('Demand data missing for some indices.')
m = gp.Model('FDK57_Car_Dealership')
x = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(x[i] <= inventory[i], name=f'inv_{i}')
    m.addConstr(x[i] <= demand[i], name=f'dem_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')