import gurobipy as gp
from gurobipy import GRB
areas = ['Queens', 'Brooklyn', 'Manhattan', 'Bronx', 'Staten Island', 'Harlem', 'Upper East Side', 'Lower Manhattan', 'Midtown', 'Long Island City', 'Williamsburg', 'Bushwick', 'Flatbush', 'Greenpoint', 'Park Slope', 'Astoria', 'Jackson Heights', 'Flushing', 'Sunnyside', 'Ditmars']
value = {'Queens': 443, 'Brooklyn': 522, 'Manhattan': 300, 'Bronx': 767, 'Staten Island': 300, 'Harlem': 309, 'Upper East Side': 598, 'Lower Manhattan': 460, 'Midtown': 318, 'Long Island City': 126, 'Williamsburg': 593, 'Bushwick': 871, 'Flatbush': 858, 'Greenpoint': 321, 'Park Slope': 275, 'Astoria': 700, 'Jackson Heights': 685, 'Flushing': 940, 'Sunnyside': 522, 'Ditmars': 763}
weight = {'Queens': 104, 'Brooklyn': 368, 'Manhattan': 483, 'Bronx': 165, 'Staten Island': 105, 'Harlem': 123, 'Upper East Side': 131, 'Lower Manhattan': 341, 'Midtown': 258, 'Long Island City': 469, 'Williamsburg': 387, 'Bushwick': 425, 'Flatbush': 482, 'Greenpoint': 495, 'Park Slope': 305, 'Astoria': 377, 'Jackson Heights': 318, 'Flushing': 56, 'Sunnyside': 213, 'Ditmars': 472}
capacity = 4466
if set(value.keys()) != set(areas) or set(weight.keys()) != set(areas):
    raise ValueError('Mismatch between area keys and value/weight keys.')
m = gp.Model('NYC_RealEstate_Dev')
x = m.addVars(areas, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[a] * x[a] for a in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[a] * x[a] for a in areas)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')