import gurobipy as gp
from gurobipy import GRB
suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
demand = {'C1': 216, 'C2': 168, 'C3': 264, 'C4': 216, 'C5': 216, 'C6': 192, 'C7': 144, 'C8': 168, 'C9': 168, 'C10': 168}
supply_capacity = {'S1': 288, 'S2': 288, 'S3': 264, 'S4': 264, 'S5': 216, 'S6': 216, 'S7': 168, 'S8': 216, 'S9': 240, 'S10': 168}
cost = {'S1': {'C1': 590.36481365, 'C2': 23.66917261, 'C3': 88.8900587, 'C4': 497.52228807, 'C5': 466.09034322, 'C6': 29.02209683, 'C7': 23.67524483, 'C8': 23.67776029, 'C9': 0.31183949, 'C10': 58.89547392}, 'S2': {'C1': 2042.0715002, 'C2': 2133.9784843, 'C3': 705.15912034, 'C4': 101.59454516, 'C5': 2052.9376574, 'C6': 1738.7549514, 'C7': 101.61094966, 'C8': 101.61062174, 'C9': 122.45214269, 'C10': 67.29170751}, 'S3': {'C1': 22.29722216, 'C2': 497.92719393, 'C3': 1653.0828862, 'C4': 23.68545123, 'C5': 1386.0807887, 'C6': 26.13715281, 'C7': 497.62204829, 'C8': 498.09358471, 'C9': 865.38162963, 'C10': 1008.6717395}, 'S4': {'C1': 960.78145339, 'C2': 49.12830005, 'C3': 1324.2386971, 'C4': 1032.2095478, 'C5': 0.07804725, 'C6': 53.30826873, 'C7': 49.13641672, 'C8': 1031.8214425, 'C9': 466.00495308, 'C10': 1351.8189071}, 'S5': {'C1': 1471.2721666, 'C2': 85.69560728, 'C3': 38.89266824, 'C4': 1542.0500358, 'C5': 112.20514372, 'C6': 82.37020164, 'C7': 1542.3399197, 'C8': 85.69238746, 'C9': 1924.9360769, 'C10': 1094.6695961}, 'S6': {'C1': 191.90587261, 'C2': 158.50401032, 'C3': 91.0204535, 'C4': 184.44747202, 'C5': 968.1467987, 'C6': 284.10760621, 'C7': 8.79106159, 'C8': 158.70523836, 'C9': 27.94387435, 'C10': 929.80716828}, 'S7': {'C1': 81.23891457, 'C2': 0.37446422, 'C3': 2079.4668654, 'C4': 0.30656718, 'C5': 1031.7772962, 'C6': 7.20396449, 'C7': 0.07623072, 'C8': 0.03247388, 'C9': 23.68582797, 'C10': 849.97994066}, 'S8': {'C1': 56.09931097, 'C2': 935.61431087, 'C3': 73.08824617, 'C4': 52.00392409, 'C5': 4.02579239, 'C6': 1002.2327658, 'C7': 935.77660296, 'C8': 935.70072523, 'C9': 612.86987193, 'C10': 1348.8366146}, 'S9': {'C1': 4.50228333, 'C2': 0.38995853, 'C3': 1782.4662178, 'C4': 0.00634591, 'C5': 1031.9910115, 'C6': 129.5066562, 'C7': 0.21183196, 'C8': 0.64573011, 'C9': 497.62723911, 'C10': 40.46575555}, 'S10': {'C1': 333.68692704, 'C2': 277.47193861, 'C3': 86.02096892, 'C4': 277.30836609, 'C5': 1004.4649085, 'C6': 19.95033682, 'C7': 13.20207369, 'C8': 238.14321522, 'C9': 411.05803324, 'C10': 941.75263656}}
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost entry for supplier {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
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