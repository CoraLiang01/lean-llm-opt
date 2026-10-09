import gurobipy as gp
from gurobipy import GRB
warehouses = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
customer_demand = {'C1': 45, 'C2': 23, 'C3': 94, 'C4': 92, 'C5': 57, 'C6': 52, 'C7': 23, 'C8': 99, 'C9': 99, 'C10': 77}
supply_capacity = {'S1': 127, 'S2': 236, 'S3': 168, 'S4': 115, 'S5': 280, 'S6': 179, 'S7': 135, 'S8': 263, 'S9': 283, 'S10': 476}
transportation_costs = {'S1': {'C1': 2077.0586725, 'C2': 0.0, 'C3': 54.3352648, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629, 'C7': 0.0, 'C8': 0.0, 'C9': 169.3302693, 'C10': 0.0}, 'S2': {'C1': 2077.0586725, 'C2': 0.0, 'C3': 1141.0405609, 'C4': 0.0, 'C5': 0.0, 'C6': 651.1112332, 'C7': 0.0, 'C8': 0.0, 'C9': 8.06334616, 'C10': 0.0}, 'S3': {'C1': 79.9210296, 'C2': 474.2450913, 'C3': 1477.0676289, 'C4': 22.58309959, 'C5': 474.2450913, 'C6': 41.10659696, 'C7': 474.2450913, 'C8': 474.2450913, 'C9': 624.1625395, 'C10': 474.2450913}, 'S4': {'C1': 1659.3369291, 'C2': 57.20541469, 'C3': 186.15190481, 'C4': 1201.3137084, 'C5': 1029.6974644, 'C6': 41.82210594, 'C7': 57.20541469, 'C8': 1201.3137084, 'C9': 884.5633871, 'C10': 1029.6974644}, 'S5': {'C1': 1297.2567041, 'C2': 77.76629131, 'C3': 24.26760228, 'C4': 1399.7932436, 'C5': 77.76629131, 'C6': 53.91161728, 'C7': 1399.7932436, 'C8': 77.76629131, 'C9': 1255.115148, 'C10': 1399.7932436}, 'S6': {'C1': 1998.9090659, 'C2': 985.3165436, 'C3': 2.854168689, 'C4': 1149.5359675, 'C5': 985.3165436, 'C6': 730.6923648, 'C7': 54.73980798, 'C8': 985.3165436, 'C9': 46.80310221, 'C10': 1149.5359675}, 'S7': {'C1': 1780.336005, 'C2': 0.0, 'C3': 1141.0405609, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629, 'C7': 0.0, 'C8': 0.0, 'C9': 8.06334616, 'C10': 0.0}, 'S8': {'C1': 75.40935896, 'C2': 1338.1987291, 'C3': 21.39134599, 'C4': 74.34437384, 'C5': 74.34437384, 'C6': 937.3506239, 'C7': 1338.1987291, 'C8': 1338.1987291, 'C9': 1392.1186581, 'C10': 1338.1987291}, 'S9': {'C1': 98.90755583, 'C2': 0.0, 'C3': 978.0347665, 'C4': 0.0, 'C5': 0.0, 'C6': 651.1112332, 'C7': 0.0, 'C8': 0.0, 'C9': 169.3302693, 'C10': 0.0}, 'S10': {'C1': 2077.0586725, 'C2': 0.0, 'C3': 54.3352648, 'C4': 0.0, 'C5': 0.0, 'C6': 36.17284629, 'C7': 0.0, 'C8': 0.0, 'C9': 145.1402308, 'C10': 0.0}}
for i in warehouses:
    if i not in supply_capacity:
        raise ValueError(f'Missing supply capacity for warehouse {i}')
    if i not in transportation_costs:
        raise ValueError(f'Missing transportation costs for warehouse {i}')
    for j in customers:
        if j not in transportation_costs[i]:
            raise ValueError(f'Missing transportation cost for warehouse {i} to customer {j}')
for j in customers:
    if j not in customer_demand:
        raise ValueError(f'Missing demand for customer {j}')
m = gp.Model()
x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((transportation_costs[i][j] * x_vars[i, j] for i in warehouses for j in customers)), GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in warehouses)) == customer_demand[j], name='')
for i in warehouses:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in customers)) <= supply_capacity[i], name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for i in warehouses:
        for j in customers:
            v = x_vars[i, j]
            print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')