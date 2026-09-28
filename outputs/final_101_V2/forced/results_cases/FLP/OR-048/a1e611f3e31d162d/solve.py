import gurobipy as gp
from gurobipy import GRB
storage_areas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
product_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
capacities = {1: 1083, 2: 1840, 3: 770, 4: 1299, 5: 1259, 6: 543, 7: 1831, 8: 855, 9: 619, 10: 637, 11: 935, 12: 626, 13: 1457, 14: 1198, 15: 837}
values = {1: 4811, 2: 1130, 3: 1611, 4: 3368, 5: 2135, 6: 1046, 7: 4030, 8: 3761, 9: 3523, 10: 1701}
weights = {1: 114, 2: 200, 3: 106, 4: 256, 5: 268, 6: 185, 7: 299, 8: 131, 9: 139, 10: 105}
if set(storage_areas) != set(capacities.keys()):
    raise ValueError('Mismatch between storage_areas and capacities keys')
if set(product_types) != set(values.keys()):
    raise ValueError('Mismatch between product_types and values keys')
if set(product_types) != set(weights.keys()):
    raise ValueError('Mismatch between product_types and weights keys')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storage_areas, product_types, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in storage_areas for j in product_types)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in product_types)) <= capacities[i] for i in storage_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')