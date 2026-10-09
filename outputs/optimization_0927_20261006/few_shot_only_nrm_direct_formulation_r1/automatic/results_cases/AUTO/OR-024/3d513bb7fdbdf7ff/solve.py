import gurobipy as gp
from gurobipy import GRB
products = ['S700_1138', 'S700_1691', 'S700_1938', 'S700_2047', 'S700_2466', 'S700_2610', 'S700_2824', 'S700_2834', 'S700_3167', 'S700_3505', 'S700_3962', 'S700_4002']
revenue = {'S700_1138': 70.67, 'S700_1691': 100.0, 'S700_1938': 70.15, 'S700_2047': 100.0, 'S700_2466': 100.0, 'S700_2610': 65.77, 'S700_2824': 100.0, 'S700_2834': 100.0, 'S700_3167': 74.4, 'S700_3505': 81.14, 'S700_3962': 100.0, 'S700_4002': 61.44}
demand = {'S700_1138': 1219, 'S700_1691': 1127, 'S700_1938': 1129, 'S700_2047': 1176, 'S700_2466': 1301, 'S700_2610': 1340, 'S700_2824': 1357, 'S700_2834': 1158, 'S700_3167': 1287, 'S700_3505': 1281, 'S700_3962': 1135, 'S700_4002': 1392}
inventory = {'S700_1138': 9020, 'S700_1691': 8370, 'S700_1938': 8390, 'S700_2047': 8680, 'S700_2466': 9400, 'S700_2610': 9900, 'S700_2824': 9760, 'S700_2834': 8610, 'S700_3167': 9380, 'S700_3505': 9170, 'S700_3962': 8520, 'S700_4002': 10290}
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing data for product {p}')
m = gp.Model('retail_revenue_max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= inventory[p] for p in products), name='')
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')