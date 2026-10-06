import gurobipy as gp
from gurobipy import GRB

def solve_transportation():
    suppliers = ['supplier1', 'supplier2', 'supplier3', 'supplier4', 'supplier5', 'supplier6', 'supplier7', 'supplier8']
    customers = ['demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8']
    supply_capacity = {'supplier1': 60, 'supplier2': 22, 'supplier3': 16, 'supplier4': 14, 'supplier5': 19, 'supplier6': 70, 'supplier7': 60, 'supplier8': 39}
    customer_demand = {'demand1': 9, 'demand2': 66, 'demand3': 56, 'demand4': 17, 'demand5': 43, 'demand6': 62, 'demand7': 10, 'demand8': 37}
    transportation_costs = {'supplier1': {'demand1': 0.0302073664, 'demand2': 229.50723505, 'demand3': 198.62356558, 'demand4': 12.99505064, 'demand5': 211.20732124, 'demand6': 134.9442985, 'demand7': 9.8222063988, 'demand8': 11.394077543}, 'supplier2': {'demand1': 232.34691308, 'demand2': 3.6258726438, 'demand3': 0.2860543415, 'demand4': 45.73127693, 'demand5': 2.8304796563, 'demand6': 107.05891033, 'demand7': 299.96317913, 'demand8': 23.799354363}, 'supplier3': {'demand1': 11.061938334, 'demand2': 0.2041995327, 'demand3': 0.2789447278, 'demand4': 45.72191272, 'demand5': 59.548955657, 'demand6': 5.0975367396, 'demand7': 300.00118415, 'demand8': 23.711282708}, 'supplier4': {'demand1': 235.17948357, 'demand2': 43.794668963, 'demand3': 40.709846783, 'demand4': 0.0777449662, 'demand5': 4.2377281834, 'demand6': 131.70915517, 'demand7': 296.55587568, 'demand8': 29.810940018}, 'supplier5': {'demand1': 211.85808746, 'demand2': 47.601808765, 'demand3': 50.040077162, 'demand4': 86.14548807, 'demand5': 0.0619789792, 'demand6': 5.3345515296, 'demand7': 270.06290424, 'demand8': 3.853933134}, 'supplier6': {'demand1': 6.455066336, 'demand2': 88.163236234, 'demand3': 5.0470916716, 'demand4': 151.46120287, 'demand5': 5.2907601611, 'demand6': 0.0460220534, 'demand7': 9.9367066018, 'demand8': 103.75460989}, 'supplier7': {'demand1': 174.27229047, 'demand2': 250.58223529, 'demand3': 253.90413042, 'demand4': 16.23546732, 'demand5': 12.643140515, 'demand6': 175.06728241, 'demand7': 2.9838396253, 'demand8': 317.06551939}, 'supplier8': {'demand1': 207.87006254, 'demand2': 1.5171684715, 'demand3': 24.027239288, 'demand4': 27.13399928, 'demand5': 73.206724689, 'demand6': 125.7291036, 'demand7': 15.463103252, 'demand8': 0.2016498751}}
    for i in suppliers:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for {i}')
        if i not in transportation_costs:
            raise ValueError(f'Missing transportation costs for {i}')
        for j in customers:
            if j not in transportation_costs[i]:
                raise ValueError(f'Missing transportation cost for ({i}, {j})')
    for j in customers:
        if j not in customer_demand:
            raise ValueError(f'Missing demand for {j}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((transportation_costs[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == customer_demand[j], name='')
    for i in suppliers:
        m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in suppliers:
            for j in customers:
                var = x[i, j]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_transportation()