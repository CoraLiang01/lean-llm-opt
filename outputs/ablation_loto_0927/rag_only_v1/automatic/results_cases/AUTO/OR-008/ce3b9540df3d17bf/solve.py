import gurobipy as gp
from gurobipy import GRB

def solve_freshmart_transportation():
    suppliers = ['Supplier1', 'Supplier2', 'Supplier3', 'Supplier4', 'Supplier5']
    customers = ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
    demand = {'Customer1': 70, 'Customer2': 80, 'Customer3': 60, 'Customer4': 90, 'Customer5': 85, 'Customer6': 95}
    supply_capacity = {'Supplier1': 200, 'Supplier2': 250, 'Supplier3': 230, 'Supplier4': 220, 'Supplier5': 210}
    transportation_cost = {('Supplier1', 'Customer1'): 2, ('Supplier1', 'Customer2'): 3, ('Supplier1', 'Customer3'): 1, ('Supplier1', 'Customer4'): 2, ('Supplier1', 'Customer5'): 3, ('Supplier1', 'Customer6'): 2, ('Supplier2', 'Customer1'): 1, ('Supplier2', 'Customer2'): 2, ('Supplier2', 'Customer3'): 3, ('Supplier2', 'Customer4'): 2, ('Supplier2', 'Customer5'): 3, ('Supplier2', 'Customer6'): 2, ('Supplier3', 'Customer1'): 3, ('Supplier3', 'Customer2'): 1, ('Supplier3', 'Customer3'): 2, ('Supplier3', 'Customer4'): 3, ('Supplier3', 'Customer5'): 2, ('Supplier3', 'Customer6'): 3, ('Supplier4', 'Customer1'): 2, ('Supplier4', 'Customer2'): 3, ('Supplier4', 'Customer3'): 2, ('Supplier4', 'Customer4'): 1, ('Supplier4', 'Customer5'): 3, ('Supplier4', 'Customer6'): 4, ('Supplier5', 'Customer1'): 3, ('Supplier5', 'Customer2'): 2, ('Supplier5', 'Customer3'): 3, ('Supplier5', 'Customer4'): 3, ('Supplier5', 'Customer5'): 2, ('Supplier5', 'Customer6'): 3}
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    for i in suppliers:
        if i not in supply_capacity:
            raise ValueError(f'Missing supply capacity for supplier {i}')
    for i in suppliers:
        for j in customers:
            if (i, j) not in transportation_cost:
                raise ValueError(f'Missing transportation cost for ({i}, {j})')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.quicksum((transportation_cost[i, j] * x[i, j] for i in suppliers for j in customers))
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
                var = x[i, j]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_freshmart_transportation()