import gurobipy as gp
from gurobipy import GRB
areas = ['Queens', 'Brooklyn', 'Manhattan', 'Bronx', 'Staten Island', 'Harlem', 'Upper East Side', 'Lower Manhattan', 'Midtown', 'Long Island City', 'Williamsburg', 'Bushwick', 'Flatbush', 'Greenpoint', 'Park Slope', 'Astoria', 'Jackson Heights', 'Flushing', 'Sunnyside', 'Ditmars']
benefit_coefficients = {'Queens': 443, 'Brooklyn': 522, 'Manhattan': 300, 'Bronx': 767, 'Staten Island': 300, 'Harlem': 309, 'Upper East Side': 598, 'Lower Manhattan': 460, 'Midtown': 318, 'Long Island City': 126, 'Williamsburg': 593, 'Bushwick': 871, 'Flatbush': 858, 'Greenpoint': 321, 'Park Slope': 275, 'Astoria': 700, 'Jackson Heights': 685, 'Flushing': 940, 'Sunnyside': 522, 'Ditmars': 763}
weights = {'Queens': 104, 'Brooklyn': 368, 'Manhattan': 483, 'Bronx': 165, 'Staten Island': 105, 'Harlem': 123, 'Upper East Side': 131, 'Lower Manhattan': 341, 'Midtown': 258, 'Long Island City': 469, 'Williamsburg': 387, 'Bushwick': 425, 'Flatbush': 482, 'Greenpoint': 495, 'Park Slope': 305, 'Astoria': 377, 'Jackson Heights': 318, 'Flushing': 56, 'Sunnyside': 213, 'Ditmars': 472}
capacity = 4466
if set(areas) != set(benefit_coefficients.keys()):
    raise ValueError('Mismatch between areas and benefit_coefficients keys')
if set(areas) != set(weights.keys()):
    raise ValueError('Mismatch between areas and weights keys')
m = gp.Model('NYC_RealEstate_Development')
x_vars = m.addVars(areas, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((benefit_coefficients[a] * x_vars[a] for a in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[a] * x_vars[a] for a in areas)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for a in areas:
        print(f'{x_vars[a].VarName}: {x_vars[a].X}')
else:
    print(f'Solver status: {m.Status}')