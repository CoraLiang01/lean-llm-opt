import gurobipy as gp
from gurobipy import GRB
suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
demand = {'C1': 216, 'C2': 168, 'C3': 264, 'C4': 216, 'C5': 216, 'C6': 192, 'C7': 144, 'C8': 168, 'C9': 168, 'C10': 168}
supply_capacity = {'S1': 288, 'S2': 288, 'S3': 264, 'S4': 264, 'S5': 216, 'S6': 216, 'S7': 168, 'S8': 216, 'S9': 240, 'S10': 168}
cost = {'S1': {'C1': 590.3648137, 'C2': 23.66917261, 'C3': 88.8900587, 'C4': 497.5222881, 'C5': 466.0903432, 'C6': 29.02209683, 'C7': 23.67524483, 'C8': 23.67776029, 'C9': 0.311839491, 'C10': 58.89547392}, 'S2': {'C1': 2042.0715, 'C2': 2133.978484, 'C3': 705.1591203, 'C4': 101.5945452, 'C5': 2052.937657, 'C6': 1738.754951, 'C7': 101.6109497, 'C8': 101.6106217, 'C9': 122.4521427, 'C10': 67.29170751}, 'S3': {'C1': 22.29722216, 'C2': 497.9271939, 'C3': 1653.082886, 'C4': 23.68545123, 'C5': 1386.080789, 'C6': 26.13715281, 'C7': 497.6220483, 'C8': 498.0935847, 'C9': 865.3816296, 'C10': 1008.671739}, 'S4': {'C1': 960.7814534, 'C2': 49.12830005, 'C3': 1324.238697, 'C4': 1032.209548, 'C5': 0.078047254, 'C6': 53.30826873, 'C7': 49.13641672, 'C8': 1031.821442, 'C9': 466.0049531, 'C10': 1351.818907}, 'S5': {'C1': 1471.272167, 'C2': 85.69560728, 'C3': 38.89266824, 'C4': 1542.050036, 'C5': 112.2051437, 'C6': 82.37020164, 'C7': 1542.33992, 'C8': 85.69238746, 'C9': 1924.936077, 'C10': 1094.669596}, 'S6': {'C1': 191.9058726, 'C2': 158.5040103, 'C3': 91.0204535, 'C4': 184.447472, 'C5': 968.1467987, 'C6': 284.1076062, 'C7': 8.791061588, 'C8': 158.7052384, 'C9': 27.94387435, 'C10': 929.8071683}, 'S7': {'C1': 81.23891457, 'C2': 0.374464222, 'C3': 2079.466865, 'C4': 0.306567176, 'C5': 1031.777296, 'C6': 7.203964492, 'C7': 0.076230722, 'C8': 0.03247388, 'C9': 23.68582797, 'C10': 849.9799407}, 'S8': {'C1': 56.09931097, 'C2': 935.6143109, 'C3': 73.08824617, 'C4': 52.00392409, 'C5': 4.025792389, 'C6': 1002.232766, 'C7': 935.776603, 'C8': 935.7007252, 'C9': 612.8698719, 'C10': 1348.836615}, 'S9': {'C1': 4.502283327, 'C2': 0.389958534, 'C3': 1782.466218, 'C4': 0.006345907, 'C5': 1031.991011, 'C6': 129.5066562, 'C7': 0.211831957, 'C8': 0.645730107, 'C9': 497.6272391, 'C10': 40.46575555}, 'S10': {'C1': 333.686927, 'C2': 277.4719386, 'C3': 86.02096892, 'C4': 277.3083661, 'C5': 1004.464908, 'C6': 19.95033682, 'C7': 13.20207369, 'C8': 238.1432152, 'C9': 411.0580332, 'C10': 941.7526366}}
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing cost data for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for supplier {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand data for customer {j}')
for i in suppliers:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for supplier {i}')
m = gp.Model('Original_RAG_TP')
x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')