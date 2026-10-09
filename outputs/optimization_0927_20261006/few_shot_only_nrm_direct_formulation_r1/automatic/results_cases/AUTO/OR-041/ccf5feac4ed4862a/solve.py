import gurobipy as gp
from gurobipy import GRB
areas = ['Queens', 'Brooklyn', 'Manhattan', 'Bronx', 'Staten Island', 'Harlem', 'Upper East Side', 'Lower Manhattan', 'Midtown', 'Long Island City', 'Williamsburg', 'Bushwick', 'Flatbush', 'Greenpoint', 'Park Slope', 'Astoria', 'Jackson Heights', 'Flushing', 'Sunnyside', 'Ditmars']
values = {'Queens': 469, 'Brooklyn': 290, 'Manhattan': 236, 'Bronx': 235, 'Staten Island': 745, 'Harlem': 684, 'Upper East Side': 444, 'Lower Manhattan': 172, 'Midtown': 1000, 'Long Island City': 336, 'Williamsburg': 546, 'Bushwick': 535, 'Flatbush': 539, 'Greenpoint': 831, 'Park Slope': 139, 'Astoria': 432, 'Jackson Heights': 627, 'Flushing': 629, 'Sunnyside': 292, 'Ditmars': 978}
weights = {'Queens': 954, 'Brooklyn': 650, 'Manhattan': 961, 'Bronx': 950, 'Staten Island': 379, 'Harlem': 776, 'Upper East Side': 381, 'Lower Manhattan': 808, 'Midtown': 937, 'Long Island City': 608, 'Williamsburg': 912, 'Bushwick': 391, 'Flatbush': 465, 'Greenpoint': 490, 'Park Slope': 918, 'Astoria': 787, 'Jackson Heights': 347, 'Flushing': 274, 'Sunnyside': 642, 'Ditmars': 130}
capacity = 586
if set(values.keys()) != set(areas):
    raise ValueError('Mismatch between areas and values keys')
if set(weights.keys()) != set(areas):
    raise ValueError('Mismatch between areas and weights keys')
m = gp.Model('NY_RealEstate_Development')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x_vars[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x_vars[i] for i in areas)) <= capacity, name='capacity')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')