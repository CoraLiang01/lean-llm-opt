import gurobipy as gp
from gurobipy import GRB
products = ['Aalopuri', 'Cold coffee', 'Frankie', 'Panipuri', 'Sandwich', 'Sugarcane juice', 'Vadapav']
revenue = {'Aalopuri': 20, 'Cold coffee': 40, 'Frankie': 50, 'Panipuri': 20, 'Sandwich': 60, 'Sugarcane juice': 25, 'Vadapav': 20}
demand = {'Aalopuri': 1483, 'Cold coffee': 1918, 'Frankie': 1623, 'Panipuri': 1720, 'Sandwich': 1558, 'Sugarcane juice': 1791, 'Vadapav': 1426}
inventory = {'Aalopuri': 10440, 'Cold coffee': 13610, 'Frankie': 11500, 'Panipuri': 12260, 'Sandwich': 10970, 'Sugarcane juice': 12780, 'Vadapav': 10060}
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('Aalop_Products_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x[p] <= inventory[p] for p in products), name='')
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')