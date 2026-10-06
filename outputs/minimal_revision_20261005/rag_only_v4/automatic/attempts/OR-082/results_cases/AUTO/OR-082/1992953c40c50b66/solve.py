import pandas as pd
import gurobipy as gp
from gurobipy import GRB
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',')
locations = ['Depot', 'A', 'B', 'C']

def norm(s):
    return str(s).casefold().strip()
norm_locations = [norm(loc) for loc in locations]
row_map = {norm(row): row for row in distance_df['Unnamed: 0']}
col_map = {norm(col): col for col in distance_df.columns if col != 'Unnamed: 0'}
missing_rows = [loc for loc in norm_locations if loc not in row_map]
missing_cols = [loc for loc in norm_locations if loc not in col_map]
if missing_rows or missing_cols:
    raise ValueError(f'Missing rows: {missing_rows}, Missing columns: {missing_cols}')
distance = {}
for i in locations:
    norm_i = norm(i)
    row_label = row_map[norm_i]
    row = distance_df[distance_df['Unnamed: 0'].apply(norm) == norm_i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in distance matrix.")
    row = row.iloc[0]
    distance[i] = {}
    for j in locations:
        norm_j = norm(j)
        col_label = col_map[norm_j]
        val = row[col_label]
        if pd.isnull(val):
            raise ValueError(f'Missing distance from {i} to {j} in distance matrix.')
        distance[i][j] = float(val)
arcs = [(i, j) for i in locations for j in locations if i != j]

def solve_problem():
    m = gp.Model('TSP_Courier')
    m.Params.MIPGap = 0.0001
    x = m.addVars(arcs, vtype=GRB.BINARY, lb=0, ub=1, name='')
    m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for (i, j) in arcs)), GRB.MINIMIZE)
    for i in locations:
        m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name='depart_%s' % i)
    for j in locations:
        m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name='arrive_%s' % j)
    n = len(locations)
    u = m.addVars([loc for loc in locations if loc != 'Depot'], vtype=GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
    for i in locations:
        for j in locations:
            if i != j and i != 'Depot' and (j != 'Depot'):
                m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name='mtz_%s_%s' % (i, j))
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')