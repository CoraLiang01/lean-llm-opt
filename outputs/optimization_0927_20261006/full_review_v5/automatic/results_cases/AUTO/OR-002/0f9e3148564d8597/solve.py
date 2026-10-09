import gurobipy as gp
from gurobipy import GRB
stores = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
demand = {'C1': 11, 'C2': 1148, 'C3': 54, 'C4': 833, 'C5': 154, 'C6': 551, 'C7': 7081, 'C8': 76, 'C9': 66, 'C10': 174, 'C11': 15, 'C12': 680}
supply_capacity = {'S1': 4, 'S2': 575, 'S3': 1504, 'S4': 178, 'S5': 228, 'S6': 50, 'S7': 3, 'S8': 6148, 'S9': 6, 'S10': 10673, 'S11': 174}
cost = {'S1': {'C1': 0.6391, 'C2': 49.7184, 'C3': 33.7586, 'C4': 1570.6731, 'C5': 1370.4095, 'C6': 57.3531, 'C7': 57.183, 'C8': 54.921, 'C9': 1143.6809, 'C10': 52.4913, 'C11': 606.4434, 'C12': 1192.4687}, 'S2': {'C1': 605.4786, 'C2': 64.5356, 'C3': 478.4779, 'C4': 887.0481, 'C5': 65.4611, 'C6': 71.9361, 'C7': 41.2902, 'C8': 70.3604, 'C9': 35.3589, 'C10': 1472.7482, 'C11': 0.6005, 'C12': 49.8685}, 'S3': {'C1': 1139.044, 'C2': 4.7851, 'C3': 1805.6214, 'C4': 1302.8958, 'C5': 2437.3212, 'C6': 103.8037, 'C7': 774.6558, 'C8': 4.516, 'C9': 879.7049, 'C10': 162.7056, 'C11': 1208.6135, 'C12': 110.1869}, 'S4': {'C1': 69.2699, 'C2': 2105.4854, 'C3': 869.682, 'C4': 1494.8986, 'C5': 310.5377, 'C6': 98.1546, 'C7': 103.3692, 'C8': 1758.8784, 'C9': 97.2854, 'C10': 94.6504, 'C11': 1277.2515, 'C12': 21.6362}, 'S5': {'C1': 980.4114, 'C2': 899.3109, 'C3': 1183.0326, 'C4': 402.0986, 'C5': 81.7886, 'C6': 1115.6819, 'C7': 123.8043, 'C8': 1121.1469, 'C9': 0.0024, 'C10': 1009.6452, 'C11': 35.348, 'C12': 1625.4346}, 'S6': {'C1': 1246.7825, 'C2': 2105.7967, 'C3': 1014.3393, 'C4': 1494.6681, 'C5': 362.0174, 'C6': 98.1714, 'C7': 2170.4059, 'C8': 97.7319, 'C9': 97.2683, 'C10': 1987.9908, 'C11': 70.944, 'C12': 389.1598}, 'S7': {'C1': 57.1086, 'C2': 23.8362, 'C3': 78.1057, 'C4': 742.8068, 'C5': 1926.0797, 'C6': 454.379, 'C7': 458.2901, 'C8': 465.9308, 'C9': 28.1386, 'C10': 524.6154, 'C11': 997.5318, 'C12': 104.4779}, 'S8': {'C1': 981.2909, 'C2': 120.9013, 'C3': 1625.8207, 'C4': 1267.8229, 'C5': 2569.6446, 'C6': 13.4718, 'C7': 815.1525, 'C8': 253.4235, 'C9': 43.7656, 'C10': 275.9784, 'C11': 1228.0699, 'C12': 103.4832}, 'S9': {'C1': 30.5328, 'C2': 1444.8595, 'C3': 173.5547, 'C4': 1307.3913, 'C5': 965.2012, 'C6': 1843.7769, 'C7': 1483.6409, 'C8': 85.3221, 'C9': 1353.5009, 'C10': 1485.9154, 'C11': 29.4238, 'C12': 26.6194}, 'S10': {'C1': 94.1109, 'C2': 1422.9971, 'C3': 1470.7769, 'C4': 1419.3382, 'C5': 38.9453, 'C6': 72.2011, 'C7': 2040.4606, 'C8': 1542.7026, 'C9': 1803.8002, 'C10': 72.9437, 'C11': 2181.4542, 'C12': 973.5516}, 'S11': {'C1': 1032.9074, 'C2': 166.3018, 'C3': 1620.4767, 'C4': 64.6683, 'C5': 2000.5092, 'C6': 0.0029, 'C7': 47.0384, 'C8': 52.9922, 'C9': 1115.6336, 'C10': 129.7934, 'C11': 1295.0978, 'C12': 2330.7682}}
for i in stores:
    if i not in cost:
        raise ValueError(f'Missing cost data for store {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for store {i}, customer {j}')
for j in customers:
    if j not in demand:
        raise ValueError(f'Missing demand data for customer {j}')
for i in stores:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for store {i}')
m = gp.Model('Walmart_Transportation')
x_vars = m.addVars(stores, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in stores for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in stores)) >= demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i] for i in stores), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')