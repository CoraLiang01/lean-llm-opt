import gurobipy as gp
from gurobipy import GRB
n_items = 140
values = [10, 40, 30, 50, 35, 6, 33, 57, 42, 24, 61, 5, 52, 16, 7, 46, 78, 36, 69, 41, 33, 41, 12, 73, 40, 34, 35, 11, 66, 42, 61, 24, 38, 37, 45, 42, 23, 8, 51, 59, 6, 71, 66, 18, 31, 15, 42, 69, 21, 5, 82, 24, 44, 49, 67, 40, 11, 32, 44, 33, 5, 24, 10, 37, 52, 46, 52, 30, 69, 25, 32, 76, 31, 5, 23, 69, 12, 28, 13, 51, 44, 22, 26, 25, 47, 72, 23, 9, 55, 62, 56, 9, 32, 51, 71, 20, 41, 73, 32, 46, 20, 21, 58, 49, 12, 53, 12, 65, 8, 50, 42, 41, 54, 40, 36, 13, 56, 28, 41, 33, 28, 9, 10, 16, 10, 23, 29, 30, 24, 38, 42, 7, 48, 34, 46, 50, 50, 4, 29, 41]
weights = [2, 5, 4, 8, 7, 1, 8, 7, 5, 5, 9, 1, 7, 3, 1, 6, 10, 8, 8, 8, 8, 6, 2, 9, 5, 6, 4, 2, 10, 8, 7, 5, 9, 6, 5, 5, 3, 1, 6, 9, 1, 9, 9, 3, 7, 2, 8, 8, 4, 1, 10, 5, 9, 7, 8, 8, 2, 4, 5, 5, 1, 6, 2, 8, 7, 10, 8, 4, 10, 5, 4, 10, 4, 1, 5, 8, 2, 5, 2, 7, 5, 4, 3, 6, 7, 10, 5, 2, 9, 7, 8, 1, 4, 8, 9, 5, 9, 9, 4, 9, 3, 3, 7, 7, 2, 9, 2, 9, 1, 8, 8, 8, 7, 5, 8, 3, 8, 6, 5, 6, 6, 1, 2, 3, 2, 5, 7, 7, 5, 9, 6, 1, 8, 6, 7, 6, 6, 1, 6, 8]
if len(values) != n_items or len(weights) != n_items:
    raise ValueError('Data length mismatch: values or weights do not match n_items.')
item_ids = list(range(1, n_items + 1))
value_dict = {i: v for (i, v) in zip(item_ids, values)}
weight_dict = {i: w for (i, w) in zip(item_ids, weights)}
m = gp.Model('shopping_centre_knapsack')
x_vars = m.addVars(item_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in item_ids:
        print(f'{x_vars[i].VarName}: {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')