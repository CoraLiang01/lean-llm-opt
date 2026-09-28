import gurobipy as gp
from gurobipy import GRB
shelves = {1: 750, 2: 820, 3: 570, 4: 800, 5: 550, 6: 900, 7: 650, 8: 800, 9: 850, 10: 900}
products = {1: {'Value': 55, 'Weight': 10}, 2: {'Value': 75, 'Weight': 20}, 3: {'Value': 65, 'Weight': 5}, 4: {'Value': 60, 'Weight': 15}, 5: {'Value': 80, 'Weight': 25}, 6: {'Value': 90, 'Weight': 35}, 7: {'Value': 40, 'Weight': 45}, 8: {'Value': 100, 'Weight': 55}, 9: {'Value': 55, 'Weight': 65}, 10: {'Value': 75, 'Weight': 20}, 11: {'Value': 110, 'Weight': 18}, 12: {'Value': 50, 'Weight': 28}, 13: {'Value': 60, 'Weight': 8}, 14: {'Value': 120, 'Weight': 28}, 15: {'Value': 70, 'Weight': 25}, 16: {'Value': 110, 'Weight': 40}, 17: {'Value': 50, 'Weight': 55}, 18: {'Value': 60, 'Weight': 70}, 19: {'Value': 120, 'Weight': 85}, 20: {'Value': 100, 'Weight': 100}}
I = list(shelves.keys())
J = list(products.keys())
for i in I:
    if i not in shelves:
        raise ValueError(f'Missing shelf capacity for shelf {i}')
for j in J:
    if j not in products or 'Value' not in products[j] or 'Weight' not in products[j]:
        raise ValueError(f'Missing value/weight for product {j}')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((products[j]['Value'] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((products[j]['Weight'] * x[i, j] for j in J)) <= shelves[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')