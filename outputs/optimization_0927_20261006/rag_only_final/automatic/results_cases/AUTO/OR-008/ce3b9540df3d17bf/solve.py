import gurobipy as gp
from gurobipy import GRB

def build_and_solve():
    suppliers = ['Supplier1', 'Supplier2', 'Supplier3', 'Supplier4', 'Supplier5']
    customers = ['Customer1', 'Customer2', 'Customer3', 'Customer4', 'Customer5', 'Customer6']
    customer_demand = {'Customer1': 70, 'Customer2': 80, 'Customer3': 60, 'Customer4': 90, 'Customer5': 85, 'Customer6': 95}
    supply_capacity = {'Supplier1': 200, 'Supplier2': 250, 'Supplier3': 230, 'Supplier4': 220, 'Supplier5': 210}
    transportation_costs = {'Supplier1': {'Customer1': 2, 'Customer2': 3, 'Customer3': 1, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier2': {'Customer1': 1, 'Customer2': 2, 'Customer3': 3, 'Customer4': 2, 'Customer5': 3, 'Customer6': 2}, 'Supplier3': {'Customer1': 3, 'Customer2': 1, 'Customer3': 2, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}, 'Supplier4': {'Customer1': 2, 'Customer2': 3, 'Customer3': 2, 'Customer4': 1, 'Customer5': 3, 'Customer6': 4}, 'Supplier5': {'Customer1': 3, 'Customer2': 2, 'Customer3': 3, 'Customer4': 3, 'Customer5': 2, 'Customer6': 3}}
    if set(customer_demand.keys()) != set(customers):
        raise ValueError('Customer demand keys do not match customer set.')
    if set(supply_capacity.keys()) != set(suppliers):
        raise ValueError('Supply capacity keys do not match supplier set.')
    if set(transportation_costs.keys()) != set(suppliers):
        raise ValueError('Transportation cost supplier keys do not match supplier set.')
    for s in suppliers:
        if set(transportation_costs[s].keys()) != set(customers):
            raise ValueError(f'Transportation cost customer keys for {s} do not match customer set.')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.quicksum((transportation_costs[s][c] * x_vars[s, c] for s in suppliers for c in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((x_vars[s, c] for s in suppliers)) == customer_demand[c], name=f'demand_{c}')
    for s in suppliers:
        m.addConstr(gp.quicksum((x_vars[s, c] for c in customers)) <= supply_capacity[s], name=f'supply_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for s in suppliers:
            for c in customers:
                v = x_vars[s, c]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve()