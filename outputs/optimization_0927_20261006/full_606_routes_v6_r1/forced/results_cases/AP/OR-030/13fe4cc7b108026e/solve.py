import gurobipy as gp
from gurobipy import GRB
I = [1, 2, 3, 4, 5]
revenue = {1: 119.144, 2: 119.144, 3: 120.144, 4: 121.244, 5: 120.544}
initial_inventory = {1: 200, 2: 100, 3: 150, 4: 200, 5: 150}
demand = {1: 30, 2: 40, 3: 50, 4: 30, 5: 10}
upper_bound = {i: min(initial_inventory[i], demand[i]) for i in I}
m = gp.Model('FDK57_Fulfillment')
x_vars = m.addVars(I, lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')