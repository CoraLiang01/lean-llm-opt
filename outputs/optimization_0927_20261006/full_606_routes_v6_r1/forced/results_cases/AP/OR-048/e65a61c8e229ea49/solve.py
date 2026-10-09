import gurobipy as gp
from gurobipy import GRB
storage_areas = {'1': 1083, '2': 1840, '3': 770, '4': 1299, '5': 1259, '6': 543, '7': 1831, '8': 855, '9': 619, '10': 637, '11': 935, '12': 626, '13': 1457, '14': 1198, '15': 837}
products = [{'ProductName': 'Window Unit', 'Value': 4811, 'Weight': 114}, {'ProductName': 'Portable Unit', 'Value': 1130, 'Weight': 200}, {'ProductName': 'Split System', 'Value': 1611, 'Weight': 106}, {'ProductName': 'Ductless System', 'Value': 3368, 'Weight': 256}, {'ProductName': 'Central AC', 'Value': 2135, 'Weight': 268}, {'ProductName': 'Hybrid AC', 'Value': 1046, 'Weight': 185}, {'ProductName': 'Geothermal AC', 'Value': 4030, 'Weight': 299}, {'ProductName': 'Smart AC', 'Value': 3761, 'Weight': 131}, {'ProductName': 'Evaporative Cooler', 'Value': 3523, 'Weight': 139}, {'ProductName': 'Package Unit', 'Value': 1701, 'Weight': 105}]
storage_ids = list(storage_areas.keys())
product_ids = [str(i + 1) for i in range(len(products))]
value = {str(i + 1): products[i]['Value'] for i in range(len(products))}
weight = {str(i + 1): products[i]['Weight'] for i in range(len(products))}
m = gp.Model('Amazon_AC_Storage')
x_vars = m.addVars(storage_ids, product_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in storage_ids for j in product_ids)), GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((weight[j] * x_vars[i, j] for j in product_ids)) <= storage_areas[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')