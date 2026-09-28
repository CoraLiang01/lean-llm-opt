import gurobipy as gp
from gurobipy import GRB
storage_areas = ['Area1', 'Area2', 'Area3', 'Area4', 'Area5', 'Area6', 'Area7', 'Area8', 'Area9', 'Area10', 'Area11', 'Area12', 'Area13', 'Area14', 'Area15']
ac_types = ['Window Unit', 'Portable Unit', 'Split System', 'Ductless System', 'Central AC', 'Hybrid AC', 'Geothermal AC', 'Smart AC', 'Evaporative Cooler', 'Package Unit']
capacities = {'Area1': 1083, 'Area2': 1840, 'Area3': 770, 'Area4': 1299, 'Area5': 1259, 'Area6': 543, 'Area7': 1831, 'Area8': 855, 'Area9': 619, 'Area10': 637, 'Area11': 935, 'Area12': 626, 'Area13': 1457, 'Area14': 1198, 'Area15': 837}
values = {'Window Unit': 4811, 'Portable Unit': 1130, 'Split System': 1611, 'Ductless System': 3368, 'Central AC': 2135, 'Hybrid AC': 1046, 'Geothermal AC': 4030, 'Smart AC': 3761, 'Evaporative Cooler': 3523, 'Package Unit': 1701}
weights = {'Window Unit': 114, 'Portable Unit': 200, 'Split System': 106, 'Ductless System': 256, 'Central AC': 268, 'Hybrid AC': 185, 'Geothermal AC': 299, 'Smart AC': 131, 'Evaporative Cooler': 139, 'Package Unit': 105}
if set(capacities.keys()) != set(storage_areas):
    raise ValueError('Mismatch in storage area identifiers between capacities and storage_areas.')
if set(values.keys()) != set(ac_types) or set(weights.keys()) != set(ac_types):
    raise ValueError('Mismatch in AC type identifiers between values/weights and ac_types.')
m = gp.Model('Amazon_AC_Storage')
x = m.addVars(storage_areas, ac_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in storage_areas for j in ac_types)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in ac_types)) <= capacities[i] for i in storage_areas), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')