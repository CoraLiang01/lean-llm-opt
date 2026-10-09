import gurobipy as gp
from gurobipy import GRB
product = 'Baby Food_255.28'
revenue = {product: 255.28}
demand = {product: 3066513}
initial_inventory = {product: 22749210}
if set(revenue.keys()) != set(demand.keys()) or set(revenue.keys()) != set(initial_inventory.keys()):
    raise ValueError('Mismatch in product keys among revenue, demand, and initial_inventory.')
m = gp.Model('Baby_Food_Revenue_Maximization')
x_vars = m.addVars([product], lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in [product])), GRB.MAXIMIZE)
m.addConstr(x_vars[product] <= initial_inventory[product], name='inv_limit')
m.addConstr(x_vars[product] <= demand[product], name='demand_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')