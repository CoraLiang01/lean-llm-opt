import gurobipy as gp
from gurobipy import GRB
storage_areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
products = ['Window Unit', 'Portable Unit', 'Split System', 'Ductless System', 'Central AC', 'Hybrid AC', 'Geothermal AC', 'Smart AC', 'Evaporative Cooler', 'Package Unit']
capacities = {1: 1083, 2: 1840, 3: 770, 4: 1299, 5: 1259, 6: 543, 7: 1831, 8: 855, 9: 619, 10: 637, 11: 935, 12: 626, 13: 1457, 14: 1198, 15: 837}
values = {'Window Unit': 4811, 'Portable Unit': 1130, 'Split System': 1611, 'Ductless System': 3368, 'Central AC': 2135, 'Hybrid AC': 1046, 'Geothermal AC': 4030, 'Smart AC': 3761, 'Evaporative Cooler': 3523, 'Package Unit': 1701}
weights = {'Window Unit': 114, 'Portable Unit': 200, 'Split System': 106, 'Ductless System': 256, 'Central AC': 268, 'Hybrid AC': 185, 'Geothermal AC': 299, 'Smart AC': 131, 'Evaporative Cooler': 139, 'Package Unit': 105}
for i in storage_areas:
    if i not in capacities:
        raise ValueError(f'Missing capacity for storage area {i}')
for j in products:
    if j not in values:
        raise ValueError(f'Missing value for product {j}')
    if j not in weights:
        raise ValueError(f'Missing weight for product {j}')
m = gp.Model('Amazon_AC_Storage')
x_vars = m.addVars(storage_areas, products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[j] * x_vars[i, j] for i in storage_areas for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x_vars[i, j] for j in products)) <= capacities[i] for i in storage_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')