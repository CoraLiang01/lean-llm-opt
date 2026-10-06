import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    stores = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
    customers = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12']
    customer_demand = {'C1': 11, 'C2': 1148, 'C3': 54, 'C4': 833, 'C5': 154, 'C6': 551, 'C7': 7081, 'C8': 76, 'C9': 66, 'C10': 174, 'C11': 15, 'C12': 680}
    supply_capacity = {'S1': 4, 'S2': 575, 'S3': 1504, 'S4': 178, 'S5': 228, 'S6': 50, 'S7': 3, 'S8': 6148, 'S9': 6, 'S10': 10673, 'S11': 174}
    transportation_costs = {'S1': {'C1': 7, 'C2': 9, 'C3': 8, 'C4': 6, 'C5': 7, 'C6': 8, 'C7': 9, 'C8': 7, 'C9': 8, 'C10': 6, 'C11': 7, 'C12': 8}, 'S2': {'C1': 6, 'C2': 8, 'C3': 7, 'C4': 5, 'C5': 6, 'C6': 7, 'C7': 8, 'C8': 6, 'C9': 7, 'C10': 5, 'C11': 6, 'C12': 7}, 'S3': {'C1': 8, 'C2': 7, 'C3': 6, 'C4': 8, 'C5': 7, 'C6': 6, 'C7': 7, 'C8': 8, 'C9': 7, 'C10': 6, 'C11': 7, 'C12': 8}, 'S4': {'C1': 9, 'C2': 6, 'C3': 8, 'C4': 7, 'C5': 8, 'C6': 9, 'C7': 6, 'C8': 7, 'C9': 8, 'C10': 9, 'C11': 6, 'C12': 7}, 'S5': {'C1': 7, 'C2': 8, 'C3': 9, 'C4': 6, 'C5': 7, 'C6': 8, 'C7': 9, 'C8': 6, 'C9': 7, 'C10': 8, 'C11': 9, 'C12': 6}, 'S6': {'C1': 8, 'C2': 7, 'C3': 6, 'C4': 8, 'C5': 7, 'C6': 6, 'C7': 8, 'C8': 7, 'C9': 6, 'C10': 8, 'C11': 7, 'C12': 6}, 'S7': {'C1': 9, 'C2': 8, 'C3': 7, 'C4': 9, 'C5': 8, 'C6': 7, 'C7': 9, 'C8': 8, 'C9': 7, 'C10': 9, 'C11': 8, 'C12': 7}, 'S8': {'C1': 6, 'C2': 7, 'C3': 8, 'C4': 6, 'C5': 7, 'C6': 8, 'C7': 6, 'C8': 7, 'C9': 8, 'C10': 6, 'C11': 7, 'C12': 8}, 'S9': {'C1': 7, 'C2': 6, 'C3': 7, 'C4': 8, 'C5': 7, 'C6': 6, 'C7': 7, 'C8': 8, 'C9': 7, 'C10': 6, 'C11': 7, 'C12': 8}, 'S10': {'C1': 8, 'C2': 9, 'C3': 8, 'C4': 7, 'C5': 8, 'C6': 9, 'C7': 8, 'C8': 7, 'C9': 8, 'C10': 9, 'C11': 8, 'C12': 7}, 'S11': {'C1': 6, 'C2': 7, 'C3': 6, 'C4': 7, 'C5': 6, 'C6': 7, 'C7': 6, 'C8': 7, 'C9': 6, 'C10': 7, 'C11': 6, 'C12': 7}}
    if set(customer_demand.keys()) != set(customers):
        raise ValueError('Customer demand keys do not match customer set')
    if set(supply_capacity.keys()) != set(stores):
        raise ValueError('Supply capacity keys do not match store set')
    if set(transportation_costs.keys()) != set(stores):
        raise ValueError('Transportation cost store keys do not match store set')
    for i in stores:
        if set(transportation_costs[i].keys()) != set(customers):
            raise ValueError(f'Transportation cost customer keys for {i} do not match customer set')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(stores, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((transportation_costs[i][j] * x[i, j] for i in stores for j in customers)), GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x[i, j] for i in stores)) == customer_demand[j], name='')
    for i in stores:
        m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in stores:
            for j in customers:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()