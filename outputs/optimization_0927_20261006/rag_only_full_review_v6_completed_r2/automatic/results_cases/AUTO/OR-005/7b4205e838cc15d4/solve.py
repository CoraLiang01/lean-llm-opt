import gurobipy as gp
from gurobipy import GRB
suppliers = ['supplier1', 'supplier2', 'supplier3', 'supplier4', 'supplier5', 'supplier6', 'supplier7', 'supplier8']
demands = ['demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8']
demand_d = {'demand1': 9, 'demand2': 66, 'demand3': 56, 'demand4': 17, 'demand5': 43, 'demand6': 62, 'demand7': 10, 'demand8': 37}
supply_capacity_s = {'supplier1': 60, 'supplier2': 22, 'supplier3': 16, 'supplier4': 14, 'supplier5': 19, 'supplier6': 70, 'supplier7': 60, 'supplier8': 39}
cost_sd = {('supplier1', 'demand1'): 0.0302073664, ('supplier1', 'demand2'): 229.50723505, ('supplier1', 'demand3'): 198.62356558, ('supplier1', 'demand4'): 12.99505064, ('supplier1', 'demand5'): 211.20732124, ('supplier1', 'demand6'): 134.9442985, ('supplier1', 'demand7'): 9.822206399, ('supplier1', 'demand8'): 11.39407754, ('supplier2', 'demand1'): 232.34691308, ('supplier2', 'demand2'): 3.625872644, ('supplier2', 'demand3'): 0.2860543415, ('supplier2', 'demand4'): 45.73127693, ('supplier2', 'demand5'): 2.830479656, ('supplier2', 'demand6'): 107.05891033, ('supplier2', 'demand7'): 299.96317913, ('supplier2', 'demand8'): 23.79935436, ('supplier3', 'demand1'): 11.06193833, ('supplier3', 'demand2'): 0.204199533, ('supplier3', 'demand3'): 0.278944728, ('supplier3', 'demand4'): 45.72191272, ('supplier3', 'demand5'): 59.54895566, ('supplier3', 'demand6'): 5.09753674, ('supplier3', 'demand7'): 300.00118415, ('supplier3', 'demand8'): 23.71128271, ('supplier4', 'demand1'): 235.17948357, ('supplier4', 'demand2'): 43.79466896, ('supplier4', 'demand3'): 40.70984678, ('supplier4', 'demand4'): 0.0777449662, ('supplier4', 'demand5'): 4.237728183, ('supplier4', 'demand6'): 131.70915517, ('supplier4', 'demand7'): 296.55587568, ('supplier4', 'demand8'): 29.81094002, ('supplier5', 'demand1'): 211.85808746, ('supplier5', 'demand2'): 47.60180877, ('supplier5', 'demand3'): 50.04007716, ('supplier5', 'demand4'): 86.14548807, ('supplier5', 'demand5'): 0.0619789792, ('supplier5', 'demand6'): 5.33455153, ('supplier5', 'demand7'): 270.06290424, ('supplier5', 'demand8'): 3.853933134, ('supplier6', 'demand1'): 6.455066336, ('supplier6', 'demand2'): 88.16323623, ('supplier6', 'demand3'): 5.047091672, ('supplier6', 'demand4'): 151.46120287, ('supplier6', 'demand5'): 5.290760161, ('supplier6', 'demand6'): 0.0460220534, ('supplier6', 'demand7'): 9.936706602, ('supplier6', 'demand8'): 103.75460989, ('supplier7', 'demand1'): 174.27229047, ('supplier7', 'demand2'): 250.58223529, ('supplier7', 'demand3'): 253.90413042, ('supplier7', 'demand4'): 16.23546732, ('supplier7', 'demand5'): 12.64314051, ('supplier7', 'demand6'): 175.06728241, ('supplier7', 'demand7'): 2.983839625, ('supplier7', 'demand8'): 317.06551939, ('supplier8', 'demand1'): 207.87006254, ('supplier8', 'demand2'): 1.517168472, ('supplier8', 'demand3'): 24.02723929, ('supplier8', 'demand4'): 27.13399928, ('supplier8', 'demand5'): 73.20672469, ('supplier8', 'demand6'): 125.7291036, ('supplier8', 'demand7'): 15.46310325, ('supplier8', 'demand8'): 0.2016498751}
for s in suppliers:
    for d in demands:
        if (s, d) not in cost_sd:
            raise ValueError(f'Missing cost coefficient for ({s}, {d})')
for d in demands:
    if d not in demand_d:
        raise ValueError(f'Missing demand for {d}')
for s in suppliers:
    if s not in supply_capacity_s:
        raise ValueError(f'Missing supply capacity for {s}')
m = gp.Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(suppliers, demands, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost_sd[s, d] * x_vars[s, d] for s in suppliers for d in demands)), GRB.MINIMIZE)
for d in demands:
    m.addConstr(gp.quicksum((x_vars[s, d] for s in suppliers)) == demand_d[d], name='')
for s in suppliers:
    m.addConstr(gp.quicksum((x_vars[s, d] for d in demands)) <= supply_capacity_s[s], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for s in suppliers:
        for d in demands:
            v = x_vars[s, d]
            print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')