import gurobipy as gp
from gurobipy import GRB
storage_areas = {1: 1083, 2: 1840, 3: 770, 4: 1299, 5: 1259, 6: 543, 7: 1831, 8: 855, 9: 619, 10: 637, 11: 935, 12: 626, 13: 1457, 14: 1198, 15: 837}
products = ['Window Unit', 'Portable Unit', 'Split System', 'Ductless System', 'Central AC', 'Hybrid AC', 'Geothermal AC', 'Smart AC', 'Evaporative Cooler', 'Package Unit']
product_values = {'Window Unit': 4811, 'Portable Unit': 1130, 'Split System': 1611, 'Ductless System': 3368, 'Central AC': 2135, 'Hybrid AC': 1046, 'Geothermal AC': 4030, 'Smart AC': 3761, 'Evaporative Cooler': 3523, 'Package Unit': 1701}
product_weights = {'Window Unit': 114, 'Portable Unit': 200, 'Split System': 106, 'Ductless System': 256, 'Central AC': 268, 'Hybrid AC': 185, 'Geothermal AC': 299, 'Smart AC': 131, 'Evaporative Cooler': 139, 'Package Unit': 105}
for i in storage_areas:
    if not isinstance(storage_areas[i], (int, float)):
        raise ValueError(f'Missing or invalid capacity for storage area {i}')
for j in products:
    if j not in product_values or j not in product_weights:
        raise ValueError(f'Missing value or weight for product {j}')
m = gp.Model('Amazon_AC_Storage')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(storage_areas.keys(), products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in storage_areas for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weights[j] * x_vars[i, j] for j in products)) <= storage_areas[i] for i in storage_areas), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')