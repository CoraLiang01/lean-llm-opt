import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    facility_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
    shipping_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
    facility_df = read_csv_with_encodings(facility_path)
    shipping_df = read_csv_with_encodings(shipping_path)
    demand_df = read_csv_with_encodings(demand_path)
    I = [str(fac).strip() for fac in facility_df['Facility']]
    J = [str(dest).strip() for dest in demand_df['Destination']]
    if set(I) != set(facility_df['Facility'].astype(str).str.strip()):
        raise ValueError('Mismatch in facility identifiers between index set and facility_costs.csv')
    f_i = {str(row['Facility']).strip(): float(row['FixedCost']) for (_, row) in facility_df.iterrows()}
    u_i = {str(row['Facility']).strip(): float(row['Capacity']) for (_, row) in facility_df.iterrows()}
    if set(J) != set(demand_df['Destination'].astype(str).str.strip()):
        raise ValueError('Mismatch in distribution center identifiers between index set and demand_requirements.csv')
    d_j = {str(row['Destination']).strip(): float(row['Demand']) for (_, row) in demand_df.iterrows()}
    shipping_df['Origin'] = shipping_df['Origin'].astype(str).str.strip()
    c_ij = {}
    for (_, row) in shipping_df.iterrows():
        i = str(row['Origin']).strip()
        if i not in I:
            continue
        c_ij[i] = {}
        for j in J:
            if j not in row:
                raise ValueError(f'Shipping cost column for {j} missing in shipping_costs.csv')
            c_ij[i][j] = float(row[j])
    for i in I:
        if i not in f_i or i not in u_i or i not in c_ij:
            raise ValueError(f'Missing data for facility {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing shipping cost for ({i},{j})')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for distribution center {j}')
    m = gp.Model('ElectroTech_UFLP')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in I for j in J]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((f_i[i] * y[i] for i in I)) + gp.quicksum((c_ij[i][j] * x[i, j] for (i, j) in x_keys)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= u_i[i] * y[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()