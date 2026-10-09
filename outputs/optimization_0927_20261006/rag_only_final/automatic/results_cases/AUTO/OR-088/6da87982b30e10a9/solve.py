import gurobipy as gp
from gurobipy import GRB

def build_and_solve():
    plants = ['F%d' % i for i in range(1, 16)]
    customers = ['C%d' % j for j in range(1, 16)]
    fixed_cost = {'F1': 11250, 'F2': 13480, 'F3': 14870, 'F4': 10290, 'F5': 16740, 'F6': 13960, 'F7': 12680, 'F8': 17890, 'F9': 10950, 'F10': 15320, 'F11': 11830, 'F12': 14110, 'F13': 15970, 'F14': 13140, 'F15': 10580}
    capacity = {'F1': 101, 'F2': 124, 'F3': 139, 'F4': 86, 'F5': 157, 'F6': 133, 'F7': 118, 'F8': 162, 'F9': 92, 'F10': 144, 'F11': 107, 'F12': 129, 'F13': 151, 'F14': 113, 'F15': 85}
    demand = {'C1': 83, 'C2': 76, 'C3': 91, 'C4': 68, 'C5': 104, 'C6': 97, 'C7': 88, 'C8': 73, 'C9': 109, 'C10': 95, 'C11': 82, 'C12': 67, 'C13': 113, 'C14': 79, 'C15': 92}
    cost_matrix = [[7.8, 7.6, 6.7, 7.9, 8.1, 8.3, 7.3, 8.2, 8.1, 8.2, 7.3, 7.7, 6.7, 7.1, 7.9], [5.3, 6.0, 5.0, 6.4, 5.9, 6.2, 5.6, 6.1, 6.3, 6.1, 5.0, 5.6, 5.3, 4.9, 6.3], [7.2, 8.1, 7.4, 8.8, 8.5, 8.7, 7.7, 8.7, 8.9, 8.5, 7.2, 7.7, 7.1, 7.6, 8.4], [7.0, 7.1, 6.5, 7.9, 7.4, 7.7, 6.7, 7.9, 7.8, 7.3, 6.8, 7.0, 6.5, 6.7, 7.6], [3.5, 3.8, 2.9, 4.3, 3.6, 3.9, 3.2, 4.3, 4.5, 4.0, 3.2, 4.0, 2.9, 3.4, 3.9], [8.2, 8.6, 7.9, 9.5, 8.5, 9.3, 8.5, 9.4, 9.0, 9.2, 8.1, 8.7, 7.9, 8.5, 9.0], [6.9, 7.6, 6.8, 8.4, 8.0, 8.0, 7.6, 8.0, 8.1, 7.8, 6.9, 7.1, 7.0, 6.9, 7.5], [6.9, 7.8, 7.1, 8.7, 8.6, 8.2, 7.2, 7.9, 8.4, 7.9, 7.0, 7.4, 6.8, 7.3, 8.0], [3.5, 3.8, 2.8, 4.4, 4.2, 4.8, 3.8, 5.0, 4.5, 4.1, 3.2, 3.7, 3.7, 3.2, 4.5], [5.2, 6.1, 5.1, 6.3, 6.1, 6.0, 5.6, 6.5, 6.2, 5.9, 5.3, 6.1, 5.1, 5.2, 6.2], [5.2, 5.5, 4.5, 6.2, 5.7, 6.1, 5.1, 5.8, 5.7, 6.2, 5.2, 5.2, 4.5, 5.1, 5.4], [7.8, 8.7, 7.6, 9.0, 8.6, 9.0, 8.5, 9.3, 9.3, 8.4, 7.9, 8.2, 7.4, 7.6, 8.7], [6.7, 6.6, 6.1, 7.3, 7.1, 7.5, 6.7, 8.0, 7.6, 7.2, 6.3, 6.9, 6.2, 6.0, 7.2], [7.5, 8.6, 7.6, 8.2, 8.0, 7.9, 7.5, 8.7, 8.8, 8.1, 7.2, 7.3, 7.0, 7.0, 8.0], [5.1, 5.8, 4.6, 5.9, 6.5, 5.9, 5.2, 7.0, 7.1, 5.9, 5.1, 5.8, 5.4, 4.9, 6.0]]
    cost = {}
    for (i, plant) in enumerate(plants):
        for (j, customer) in enumerate(customers):
            cost[plant, customer] = cost_matrix[i][j]
    if set(fixed_cost.keys()) != set(plants):
        raise ValueError('Mismatch in plant identifiers for fixed_cost')
    if set(capacity.keys()) != set(plants):
        raise ValueError('Mismatch in plant identifiers for capacity')
    if set(demand.keys()) != set(customers):
        raise ValueError('Mismatch in customer identifiers for demand')
    for plant in plants:
        for customer in customers:
            if (plant, customer) not in cost:
                raise ValueError(f'Missing cost entry for ({plant}, {customer})')
    m = gp.Model()
    y_vars = m.addVars(plants, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_vars = m.addVars(plants, customers, vtype=GRB.CONTINUOUS, lb=0, name='')
    for customer in customers:
        m.addConstr(gp.quicksum((x_vars[plant, customer] for plant in plants)) == demand[customer], name='')
    for plant in plants:
        m.addConstr(gp.quicksum((x_vars[plant, customer] for customer in customers)) <= capacity[plant] * y_vars[plant], name='')
    m.setObjective(gp.quicksum((fixed_cost[plant] * y_vars[plant] for plant in plants)) + gp.quicksum((cost[plant, customer] * x_vars[plant, customer] for plant in plants for customer in customers)), GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for v in m.getVars():
            print(v.VarName, v.X)
    else:
        print('Status', m.Status)
    return m
m = build_and_solve()