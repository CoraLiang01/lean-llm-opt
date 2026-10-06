import gurobipy as gp
from gurobipy import GRB

def build_and_solve_transportation():
    suppliers = ['supply1', 'supply2', 'supply3', 'supply4', 'supply5', 'supply6', 'supply7', 'supply8']
    customers = ['demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8']
    demand = {'demand1': 9, 'demand2': 66, 'demand3': 56, 'demand4': 17, 'demand5': 43, 'demand6': 62, 'demand7': 10, 'demand8': 37}
    supply_capacity = {'supply1': 60, 'supply2': 22, 'supply3': 16, 'supply4': 14, 'supply5': 19, 'supply6': 70, 'supply7': 60, 'supply8': 39}
    cost = {'supply1': {'demand1': 0.0302073664, 'demand2': 229.50723505, 'demand3': 198.62356558, 'demand4': 12.99505064, 'demand5': 211.20732124, 'demand6': 134.9442985, 'demand7': 9.822206399, 'demand8': 11.39407754}, 'supply2': {'demand1': 232.34691308, 'demand2': 3.625872644, 'demand3': 0.2860543415, 'demand4': 45.73127693, 'demand5': 2.830479656, 'demand6': 107.05891033, 'demand7': 299.96317913, 'demand8': 23.79935436}, 'supply3': {'demand1': 11.06193833, 'demand2': 0.204199533, 'demand3': 0.278944728, 'demand4': 45.72191272, 'demand5': 59.54895566, 'demand6': 5.09753674, 'demand7': 300.00118415, 'demand8': 23.71128271}, 'supply4': {'demand1': 235.17948357, 'demand2': 43.79466896, 'demand3': 40.70984678, 'demand4': 0.0777449662, 'demand5': 4.237728183, 'demand6': 131.70915517, 'demand7': 296.55587568, 'demand8': 29.81094002}, 'supply5': {'demand1': 211.85808746, 'demand2': 47.60180877, 'demand3': 50.04007716, 'demand4': 86.14548807, 'demand5': 0.0619789792, 'demand6': 5.33455153, 'demand7': 270.06290424, 'demand8': 3.853933134}, 'supply6': {'demand1': 6.455066336, 'demand2': 88.16323623, 'demand3': 5.047091672, 'demand4': 151.46120287, 'demand5': 5.290760161, 'demand6': 0.0460220534, 'demand7': 9.936706602, 'demand8': 103.75460989}, 'supply7': {'demand1': 174.27229047, 'demand2': 250.58223529, 'demand3': 253.90413042, 'demand4': 16.23546732, 'demand5': 12.64314051, 'demand6': 175.06728241, 'demand7': 2.983839625, 'demand8': 317.06551939}, 'supply8': {'demand1': 207.87006254, 'demand2': 1.517168472, 'demand3': 24.02723929, 'demand4': 27.13399928, 'demand5': 73.20672469, 'demand6': 125.7291036, 'demand7': 15.46310325, 'demand8': 0.2016498751}}
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Missing cost data for supplier {i}')
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for supplier {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost data for supplier {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand data for customer {j}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.LinExpr()
    for i in suppliers:
        for j in customers:
            obj += cost[i][j] * x[i, j]
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name='')
    for i in suppliers:
        m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in suppliers:
            for j in customers:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve_transportation()