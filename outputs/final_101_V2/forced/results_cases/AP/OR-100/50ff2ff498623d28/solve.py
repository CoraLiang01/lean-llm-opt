import gurobipy as gp
from gurobipy import GRB
components = [f'C{i}' for i in range(1, 112)]
workshops = ['Casting', 'Milling', 'Finishing', 'Assembly', 'QA & Packaging']
unit_price = {'C1': 193, 'C2': 64, 'C3': 103, 'C4': 210, 'C5': 85, 'C6': 126, 'C7': 226, 'C8': 94, 'C9': 73, 'C10': 120, 'C111': 142}
processing_time_unit = {'C1': {'Casting': 0.74, 'Milling': 0.6, 'Finishing': 0.0, 'Assembly': 4.84, 'QA & Packaging': 0.92}, 'C2': {'Casting': 0.77, 'Milling': 3.38, 'Finishing': 4.15, 'Assembly': 0.0, 'QA & Packaging': 0.0}, 'C3': {'Casting': 1.41, 'Milling': 0.0, 'Finishing': 0.0, 'Assembly': 3.8, 'QA & Packaging': 1.08}, 'C4': {'Casting': 2.11, 'Milling': 0.0, 'Finishing': 2.45, 'Assembly': 2.01, 'QA & Packaging': 3.39}, 'C5': {'Casting': 2.19, 'Milling': 1.25, 'Finishing': 2.49, 'Assembly': 3.03, 'QA & Packaging': 2.54}, 'C6': {'Casting': 1.31, 'Milling': 2.66, 'Finishing': 1.93, 'Assembly': 2.25, 'QA & Packaging': 0.0}, 'C7': {'Casting': 3.75, 'Milling': 4.56, 'Finishing': 0.72, 'Assembly': 0.0, 'QA & Packaging': 3.44}, 'C8': {'Casting': 3.47, 'Milling': 4.81, 'Finishing': 0.0, 'Assembly': 0.95, 'QA & Packaging': 3.84}, 'C9': {'Casting': 3.8, 'Milling': 3.83, 'Finishing': 1.12, 'Assembly': 2.11, 'QA & Packaging': 2.45}, 'C10': {'Casting': 3.81, 'Milling': 2.29, 'Finishing': 2.95, 'Assembly': 1.19, 'QA & Packaging': 4.99}, 'C111': {'Casting': 3.81, 'Milling': 2.67, 'Finishing': 4.07, 'Assembly': 4.02, 'QA & Packaging': 3.21}}
total_working_hours = {'Casting': 7650, 'Milling': 6320, 'Finishing': 5538, 'Assembly': 5957, 'QA & Packaging': 6988}
for c in components:
    if c not in unit_price:
        raise ValueError(f'Missing unit price for {c}')
    if c not in processing_time_unit:
        raise ValueError(f'Missing processing time for {c}')
    for w in workshops:
        if w not in processing_time_unit[c]:
            raise ValueError(f'Missing processing time for {c} in {w}')
for w in workshops:
    if w not in total_working_hours:
        raise ValueError(f'Missing total working hours for {w}')
m = gp.Model('Factory_Production_Optimization')
x = m.addVars(components, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((unit_price[c] * x[c] for c in components)), GRB.MAXIMIZE)
for w in workshops:
    m.addConstr(gp.quicksum((processing_time_unit[c][w] * x[c] for c in components)) <= total_working_hours[w], name=f'workshop_{w}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for c in components:
        print(f'x[{c}]: {x[c].X}')
else:
    print(f'Solver status: {m.Status}')