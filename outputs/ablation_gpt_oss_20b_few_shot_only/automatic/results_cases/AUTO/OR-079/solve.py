import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    fac_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
    ship_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
    dem_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
    df_fac = pd.read_csv(fac_path, dtype=str, keep_default_na=False)
    df_ship = pd.read_csv(ship_path, dtype=str, keep_default_na=False)
    df_dem = pd.read_csv(dem_path, dtype=str, keep_default_na=False)
    facilities = df_fac['Facility'].tolist()
    f = {}
    cap = {}
    for (_, row) in df_fac.iterrows():
        fac = row['Facility']
        f[fac] = float(row['FixedCost'])
        cap[fac] = float(row['Capacity'])
    destinations = df_dem['Destination'].tolist()
    d = {}
    for (_, row) in df_dem.iterrows():
        dest = row['Destination']
        d[dest] = float(row['Demand'])
    c = {}
    for (_, row) in df_ship.iterrows():
        origin = row['Origin']
        for dest in destinations:
            c[origin, dest] = float(row[dest])
    for fac in facilities:
        if fac not in f or fac not in cap:
            raise ValueError(f'Missing cost or capacity for facility {fac}')
    for dest in destinations:
        if dest not in d:
            raise ValueError(f'Missing demand for destination {dest}')
    for fac in facilities:
        for dest in destinations:
            if (fac, dest) not in c:
                raise ValueError(f'Missing shipping cost for {fac}->{dest}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in facilities for j in destinations], lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(facilities, vtype=GRB.BINARY, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in facilities)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in facilities for j in destinations))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in facilities)) == d[j] for j in destinations), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in destinations)) <= cap[i] * y[i] for i in facilities), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')