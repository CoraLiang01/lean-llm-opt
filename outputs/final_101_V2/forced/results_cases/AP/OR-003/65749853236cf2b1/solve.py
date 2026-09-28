import gurobipy as gp
from gurobipy import GRB
suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
supply_capacity = {'S1': 288, 'S2': 288, 'S3': 264, 'S4': 264, 'S5': 216, 'S6': 216, 'S7': 168, 'S8': 216, 'S9': 240, 'S10': 168}
customer_demand = {'C1': 216, 'C2': 168, 'C3': 264, 'C4': 216, 'C5': 216, 'C6': 192, 'C7': 144, 'C8': 168, 'C9': 168, 'C10': 168}
transportation_cost = {'S1': {'C1': 590.364814, 'C2': 23.669173, 'C3': 88.890059, 'C4': 497.522288, 'C5': 466.090343, 'C6': 29.022097, 'C7': 23.675245, 'C8': 23.67776, 'C9': 0.311839, 'C10': 58.895474}, 'S2': {'C1': 2042.0715, 'C2': 2133.97848, 'C3': 705.15912, 'C4': 101.594545, 'C5': 2052.93766, 'C6': 1738.75495, 'C7': 101.61095, 'C8': 101.610622, 'C9': 122.452143, 'C10': 67.291708}, 'S3': {'C1': 22.297222, 'C2': 497.927194, 'C3': 1653.08289, 'C4': 23.685451, 'C5': 1386.08079, 'C6': 26.137153, 'C7': 497.622048, 'C8': 498.093585, 'C9': 865.38163, 'C10': 1008.67174}, 'S4': {'C1': 960.781453, 'C2': 49.1283, 'C3': 1324.2387, 'C4': 1032.20955, 'C5': 0.078047, 'C6': 53.308269, 'C7': 49.136417, 'C8': 1031.82144, 'C9': 466.004953, 'C10': 1351.81891}, 'S5': {'C1': 1471.27217, 'C2': 85.695607, 'C3': 38.892668, 'C4': 1542.05004, 'C5': 112.205144, 'C6': 82.370202, 'C7': 1542.33992, 'C8': 85.692387, 'C9': 1924.93608, 'C10': 1094.6696}, 'S6': {'C1': 191.905873, 'C2': 158.50401, 'C3': 91.020453, 'C4': 184.447472, 'C5': 968.146799, 'C6': 284.107606, 'C7': 8.791062, 'C8': 158.705238, 'C9': 27.943874, 'C10': 929.807168}, 'S7': {'C1': 81.238915, 'C2': 0.374464, 'C3': 2079.46687, 'C4': 0.306567, 'C5': 1031.7773, 'C6': 7.203964, 'C7': 0.076231, 'C8': 0.032474, 'C9': 23.685828, 'C10': 849.979941}, 'S8': {'C1': 56.099311, 'C2': 935.614311, 'C3': 73.088246, 'C4': 52.003924, 'C5': 4.025792, 'C6': 1002.23277, 'C7': 935.776603, 'C8': 935.700725, 'C9': 612.869872, 'C10': 1348.83661}, 'S9': {'C1': 4.502283, 'C2': 0.389959, 'C3': 1782.46622, 'C4': 0.006346, 'C5': 1031.99101, 'C6': 129.506656, 'C7': 0.211832, 'C8': 0.64573, 'C9': 497.627239, 'C10': 40.465756}, 'S10': {'C1': 333.686927, 'C2': 277.471939, 'C3': 86.020969, 'C4': 277.308366, 'C5': 1004.46491, 'C6': 19.950337, 'C7': 13.202074, 'C8': 238.143215, 'C9': 411.058033, 'C10': 941.752637}}
for s in suppliers:
    if s not in supply_capacity:
        raise ValueError(f'Missing supply_capacity for {s}')
    if s not in transportation_cost:
        raise ValueError(f'Missing transportation_cost for {s}')
    for c in customers:
        if c not in transportation_cost[s]:
            raise ValueError(f'Missing transportation_cost for {s},{c}')
for c in customers:
    if c not in customer_demand:
        raise ValueError(f'Missing customer_demand for {c}')
m = gp.Model('transportation')
x = m.addVars(suppliers, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
m.setObjective(gp.quicksum((transportation_cost[s][c] * x[s, c] for s in suppliers for c in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s] for s in suppliers), name='')
m.addConstrs((gp.quicksum((x[s, c] for s in suppliers)) == customer_demand[c] for c in customers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')