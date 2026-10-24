import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path_costs = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
    path_demands = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
    path_transport = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
    df_costs = pd.read_csv(path_costs, dtype=str, keep_default_na=False)
    df_demands = pd.read_csv(path_demands, dtype=str, keep_default_na=False)
    df_transport = pd.read_csv(path_transport, dtype=str, keep_default_na=False)
    df_costs['Warehouse (i)'] = df_costs['Warehouse (i)'].str.strip()
    df_costs['Opening Cost (fi)'] = df_costs['Opening Cost (fi)'].str.strip()
    df_costs['Capacity (units)'] = df_costs['Capacity (units)'].str.strip()
    f_i = {}
    cap_i = {}
    for (_, row) in df_costs.iterrows():
        key = int(row['Warehouse (i)'])
        f_i[key] = float(row['Opening Cost (fi)'])
        cap_i[key] = float(row['Capacity (units)'])
    df_demands['Store (j)'] = df_demands['Store (j)'].str.strip()
    df_demands['Demand (units, dj)'] = df_demands['Demand (units, dj)'].str.strip()
    d_j = {}
    for (_, row) in df_demands.iterrows():
        key = int(row['Store (j)'])
        d_j[key] = float(row['Demand (units, dj)'])
    df_transport.columns = df_transport.columns.str.strip()
    df_transport['Unnamed: 0'] = df_transport['Unnamed: 0'].str.strip()
    c_ij = {}
    for (_, row) in df_transport.iterrows():
        i_key = row['Unnamed: 0']
        i = int(i_key[1:])
        c_ij[i] = {}
        for col in df_transport.columns:
            if col == 'Unnamed: 0':
                continue
            j = int(col[1:])
            c_ij[i][j] = float(row[col].strip())
    I = sorted(f_i.keys())
    J = sorted(d_j.keys())
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing opening cost for warehouse {i}')
        if i not in cap_i:
            raise ValueError(f'Missing capacity for warehouse {i}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for store {j}')
    for i in I:
        if i not in c_ij:
            raise ValueError(f'Missing transportation cost row for warehouse {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i} to store {j}')
    m = gurobipy.Model('UFLP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((f_i[i] * y[i] for i in I)) + gurobipy.quicksum((c_ij[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == d_j[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= cap_i[i] * y[i], name=f'capacity_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()