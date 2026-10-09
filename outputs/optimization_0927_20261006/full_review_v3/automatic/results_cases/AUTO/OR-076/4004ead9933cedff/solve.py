import gurobipy as gp
from gurobipy import GRB
warehouses = [{'Warehouse ID': 'W1', 'Fixed_Cost': 2000, 'Capacity': 1000}, {'Warehouse ID': 'W2', 'Fixed_Cost': 2500, 'Capacity': 1500}, {'Warehouse ID': 'W3', 'Fixed_Cost': 1800, 'Capacity': 1200}, {'Warehouse ID': 'W4', 'Fixed_Cost': 3200, 'Capacity': 2000}, {'Warehouse ID': 'W5', 'Fixed_Cost': 1500, 'Capacity': 800}, {'Warehouse ID': 'W6', 'Fixed_Cost': 4000, 'Capacity': 2500}, {'Warehouse ID': 'W7', 'Fixed_Cost': 2800, 'Capacity': 1800}, {'Warehouse ID': 'W8', 'Fixed_Cost': 1950, 'Capacity': 1100}, {'Warehouse ID': 'W9', 'Fixed_Cost': 3500, 'Capacity': 2100}, {'Warehouse ID': 'W10', 'Fixed_Cost': 2200, 'Capacity': 1300}]
customers = [{'Customer ID': 'C1', 'Demand': 800}, {'Customer ID': 'C2', 'Demand': 600}, {'Customer ID': 'C3', 'Demand': 500}, {'Customer ID': 'C4', 'Demand': 700}, {'Customer ID': 'C5', 'Demand': 450}, {'Customer ID': 'C6', 'Demand': 950}, {'Customer ID': 'C7', 'Demand': 350}, {'Customer ID': 'C8', 'Demand': 850}, {'Customer ID': 'C9', 'Demand': 400}, {'Customer ID': 'C10', 'Demand': 750}, {'Customer ID': 'C11', 'Demand': 900}, {'Customer ID': 'C12', 'Demand': 550}, {'Customer ID': 'C13', 'Demand': 650}, {'Customer ID': 'C14', 'Demand': 820}, {'Customer ID': 'C15', 'Demand': 480}, {'Customer ID': 'C16', 'Demand': 920}, {'Customer ID': 'C17', 'Demand': 320}, {'Customer ID': 'C18', 'Demand': 780}, {'Customer ID': 'C19', 'Demand': 520}, {'Customer ID': 'C20', 'Demand': 680}]
cost = {'W1': {'C1': 10, 'C2': 15, 'C3': 20, 'C4': 11, 'C5': 16, 'C6': 18, 'C7': 7, 'C8': 12, 'C9': 22, 'C10': 9, 'C11': 14, 'C12': 19, 'C13': 25, 'C14': 13, 'C15': 17, 'C16': 6, 'C17': 21, 'C18': 15, 'C19': 8, 'C20': 10}, 'W2': {'C1': 18, 'C2': 12, 'C3': 9, 'C4': 14, 'C5': 10, 'C6': 5, 'C7': 19, 'C8': 23, 'C9': 11, 'C10': 16, 'C11': 20, 'C12': 8, 'C13': 15, 'C14': 22, 'C15': 7, 'C16': 13, 'C17': 24, 'C18': 17, 'C19': 12, 'C20': 6}, 'W3': {'C1': 13, 'C2': 17, 'C3': 15, 'C4': 8, 'C5': 12, 'C6': 21, 'C7': 16, 'C8': 10, 'C9': 5, 'C10': 24, 'C11': 13, 'C12': 22, 'C13': 7, 'C14': 19, 'C15': 14, 'C16': 18, 'C17': 9, 'C18': 25, 'C19': 11, 'C20': 16}, 'W4': {'C1': 7, 'C2': 22, 'C3': 11, 'C4': 16, 'C5': 20, 'C6': 8, 'C7': 15, 'C8': 19, 'C9': 13, 'C10': 25, 'C11': 6, 'C12': 14, 'C13': 21, 'C14': 9, 'C15': 23, 'C16': 17, 'C17': 10, 'C18': 18, 'C19': 24, 'C20': 5}, 'W5': {'C1': 16, 'C2': 9, 'C3': 25, 'C4': 13, 'C5': 7, 'C6': 10, 'C7': 23, 'C8': 14, 'C9': 18, 'C10': 21, 'C11': 5, 'C12': 17, 'C13': 9, 'C14': 24, 'C15': 12, 'C16': 20, 'C17': 6, 'C18': 15, 'C19': 19, 'C20': 11}, 'W6': {'C1': 22, 'C2': 6, 'C3': 14, 'C4': 19, 'C5': 23, 'C6': 11, 'C7': 8, 'C8': 17, 'C9': 9, 'C10': 12, 'C11': 15, 'C12': 24, 'C13': 5, 'C14': 20, 'C15': 10, 'C16': 25, 'C17': 13, 'C18': 7, 'C19': 18, 'C20': 16}, 'W7': {'C1': 8, 'C2': 25, 'C3': 17, 'C4': 9, 'C5': 14, 'C6': 22, 'C7': 11, 'C8': 6, 'C9': 16, 'C10': 20, 'C11': 18, 'C12': 13, 'C13': 24, 'C14': 5, 'C15': 19, 'C16': 12, 'C17': 23, 'C18': 10, 'C19': 7, 'C20': 15}, 'W8': {'C1': 19, 'C2': 11, 'C3': 7, 'C4': 21, 'C5': 15, 'C6': 24, 'C7': 13, 'C8': 16, 'C9': 20, 'C10': 8, 'C11': 17, 'C12': 10, 'C13': 12, 'C14': 23, 'C15': 5, 'C16': 14, 'C17': 22, 'C18': 9, 'C19': 16, 'C20': 25}, 'W9': {'C1': 12, 'C2': 20, 'C3': 5, 'C4': 23, 'C5': 17, 'C6': 14, 'C7': 9, 'C8': 25, 'C9': 18, 'C10': 11, 'C11': 16, 'C12': 21, 'C13': 10, 'C14': 7, 'C15': 24, 'C16': 15, 'C17': 19, 'C18': 6, 'C19': 13, 'C20': 22}, 'W10': {'C1': 25, 'C2': 14, 'C3': 22, 'C4': 5, 'C5': 19, 'C6': 12, 'C7': 24, 'C8': 7, 'C9': 15, 'C10': 17, 'C11': 23, 'C12': 6, 'C13': 16, 'C14': 10, 'C15': 20, 'C16': 9, 'C17': 18, 'C18': 11, 'C19': 25, 'C20': 14}}
warehouse_ids = [w['Warehouse ID'] for w in warehouses]
customer_ids = [c['Customer ID'] for c in customers]
fixed_cost = {w['Warehouse ID']: w['Fixed_Cost'] for w in warehouses}
capacity = {w['Warehouse ID']: w['Capacity'] for w in warehouses}
demand = {c['Customer ID']: c['Demand'] for c in customers}
for i in warehouse_ids:
    if i not in cost:
        raise ValueError(f'Missing cost data for warehouse {i}')
    for j in customer_ids:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for warehouse {i}, customer {j}')
for j in customer_ids:
    if j not in demand:
        raise ValueError(f'Missing demand for customer {j}')
for i in warehouse_ids:
    if i not in fixed_cost or i not in capacity:
        raise ValueError(f'Missing fixed cost or capacity for warehouse {i}')
m = gp.Model('Warehouse_Location')
x_vars = m.addVars(warehouse_ids, customer_ids, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouse_ids, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouse_ids)) + gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouse_ids for j in customer_ids)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouse_ids)) == demand[j] for j in customer_ids), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in customer_ids)) <= capacity[i] * y_vars[i] for i in warehouse_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')