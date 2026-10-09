import gurobipy as gp
from gurobipy import GRB
warehouses = [{'i': 1, 'f_i': 3000, 'cap_i': 180}, {'i': 2, 'f_i': 3200, 'cap_i': 160}, {'i': 3, 'f_i': 3100, 'cap_i': 200}, {'i': 4, 'f_i': 2800, 'cap_i': 150}, {'i': 5, 'f_i': 3500, 'cap_i': 170}, {'i': 6, 'f_i': 2700, 'cap_i': 190}, {'i': 7, 'f_i': 2900, 'cap_i': 160}, {'i': 8, 'f_i': 3050, 'cap_i': 175}, {'i': 9, 'f_i': 3100, 'cap_i': 170}, {'i': 10, 'f_i': 2200, 'cap_i': 180}, {'i': 11, 'f_i': 2890, 'cap_i': 190}]
stores = [{'j': 1, 'd_j': 30}, {'j': 2, 'd_j': 40}, {'j': 3, 'd_j': 20}, {'j': 4, 'd_j': 35}, {'j': 5, 'd_j': 20}, {'j': 6, 'd_j': 25}, {'j': 7, 'd_j': 45}, {'j': 8, 'd_j': 38}, {'j': 9, 'd_j': 32}, {'j': 10, 'd_j': 41}, {'j': 11, 'd_j': 44}]
transportation_costs = [[12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15], [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16], [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17], [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18], [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17], [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19], [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14], [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18], [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18], [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19], [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]]
warehouse_ids = [w['i'] for w in warehouses]
store_ids = [s['j'] for s in stores]
f_i = {w['i']: w['f_i'] for w in warehouses}
cap_i = {w['i']: w['cap_i'] for w in warehouses}
d_j = {s['j']: s['d_j'] for s in stores}
c_ij = {}
for (wi, w) in enumerate(warehouse_ids):
    for (sj, s) in enumerate(store_ids):
        c_ij[w, s] = transportation_costs[wi][sj]
if len(warehouse_ids) != 11 or len(store_ids) != 11:
    raise ValueError('Expected 11 warehouses and 11 stores.')
for w in warehouse_ids:
    if w not in f_i or w not in cap_i:
        raise ValueError(f'Missing cost or capacity for warehouse {w}.')
for s in store_ids:
    if s not in d_j:
        raise ValueError(f'Missing demand for store {s}.')
for w in warehouse_ids:
    for s in store_ids:
        if (w, s) not in c_ij:
            raise ValueError(f'Missing transportation cost for warehouse {w}, store {s}.')
m = gp.Model('Warehouse_Location')
x_vars = m.addVars(warehouse_ids, store_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouse_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((f_i[w] * y_vars[w] for w in warehouse_ids)) + gp.quicksum((c_ij[w, s] * x_vars[w, s] for w in warehouse_ids for s in store_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[w, s] for w in warehouse_ids)) == d_j[s] for s in store_ids), name='')
m.addConstrs((gp.quicksum((x_vars[w, s] for s in store_ids)) <= cap_i[w] * y_vars[w] for w in warehouse_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')