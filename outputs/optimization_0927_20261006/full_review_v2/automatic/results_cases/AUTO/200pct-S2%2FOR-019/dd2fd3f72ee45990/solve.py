import gurobipy as gp
from gurobipy import GRB
suppliers = ['supplier1', 'supplier2', 'supplier3', 'supplier4', 'supplier5', 'supplier6', 'supplier7', 'supplier8']
customers = ['demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8']
demand = {'demand1': 9, 'demand2': 66, 'demand3': 56, 'demand4': 17, 'demand5': 43, 'demand6': 62, 'demand7': 10, 'demand8': 37}
supply_capacity = {'supplier1': 60, 'supplier2': 22, 'supplier3': 16, 'supplier4': 14, 'supplier5': 19, 'supplier6': 70, 'supplier7': 60, 'supplier8': 39}
cost = {'supplier1': {'demand1': 0.0302073664, 'demand2': 229.50723505, 'demand3': 198.62356558, 'demand4': 12.99505064, 'demand5': 211.20732124, 'demand6': 134.9442985, 'demand7': 9.8222063988, 'demand8': 11.394077543}, 'supplier2': {'demand1': 232.34691308, 'demand2': 3.6258726439, 'demand3': 0.2860543415, 'demand4': 45.73127693, 'demand5': 2.8304796563, 'demand6': 107.05891033, 'demand7': 299.96317913, 'demand8': 23.799354363}, 'supplier3': {'demand1': 11.061938334, 'demand2': 0.2041995327, 'demand3': 0.2789447278, 'demand4': 45.72191272, 'demand5': 59.548955657, 'demand6': 5.0975367396, 'demand7': 300.00118415, 'demand8': 23.711282708}, 'supplier4': {'demand1': 235.17948357, 'demand2': 43.794668963, 'demand3': 40.709846783, 'demand4': 0.0777449662, 'demand5': 4.2377281834, 'demand6': 131.70915517, 'demand7': 296.55587568, 'demand8': 29.810940018}, 'supplier5': {'demand1': 211.85808746, 'demand2': 47.601808765, 'demand3': 50.040077162, 'demand4': 86.14548807, 'demand5': 0.0619789792, 'demand6': 5.3345515296, 'demand7': 270.06290424, 'demand8': 3.853933134}, 'supplier6': {'demand1': 6.4550663355, 'demand2': 88.163236234, 'demand3': 5.0470916716, 'demand4': 151.46120287, 'demand5': 5.2907601611, 'demand6': 0.0460220534, 'demand7': 9.9367066018, 'demand8': 103.75460989}, 'supplier7': {'demand1': 174.27229047, 'demand2': 250.58223529, 'demand3': 253.90413042, 'demand4': 16.235467318, 'demand5': 12.643140515, 'demand6': 175.06728241, 'demand7': 2.9838396253, 'demand8': 317.06551939}, 'supplier8': {'demand1': 207.87006254, 'demand2': 1.5171684715, 'demand3': 24.027239288, 'demand4': 27.13399928, 'demand5': 73.206724689, 'demand6': 125.7291036, 'demand7': 15.463103252, 'demand8': 0.2016498751}}
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
m = gp.Model('Amazon_Distribution_Transportation')
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