import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
    transport_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
    df_demand = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    df_fixed = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    df_transport = pd.read_csv(transport_path, dtype=str, keep_default_na=False)
    df_demand.columns = df_demand.columns.str.strip()
    df_fixed.columns = df_fixed.columns.str.strip()
    df_transport.columns = df_transport.columns.str.strip()
    df_demand['customer'] = df_demand['customer'].str.strip()
    df_fixed['Unnamed: 0'] = df_fixed['Unnamed: 0'].str.strip()
    df_transport['Unnamed: 0'] = df_transport['Unnamed: 0'].str.strip()
    I = ['S1', 'S2', 'S3', 'S4', 'S5']
    J = ['C1', 'C2', 'C3', 'C4', 'C5']
    try:
        df_demand['demand'] = df_demand['demand'].astype(float)
    except ValueError as e:
        raise ValueError(f'Invalid numeric value in demand.csv: {e}')
    try:
        df_fixed['fixed_costs'] = df_fixed['fixed_costs'].astype(float)
    except ValueError as e:
        raise ValueError(f'Invalid numeric value in fixed_cost.csv: {e}')
    for col in J:
        if col not in df_transport.columns:
            raise ValueError(f'Transportation CSV missing column {col}')
        try:
            df_transport[col] = df_transport[col].astype(float)
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in transportation_costs.csv column {col}: {e}')
    f = {}
    for (_, row) in df_fixed.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in I:
            raise ValueError(f'Unexpected supplier {supplier} in fixed_cost.csv')
        f[supplier] = float(row['fixed_costs'])
    for i in I:
        if i not in f:
            raise ValueError(f'Missing fixed cost for supplier {i}')
    d = {}
    for (_, row) in df_demand.iterrows():
        customer = row['customer']
        if customer not in J:
            raise ValueError(f'Unexpected customer {customer} in demand.csv')
        d[customer] = float(row['demand'])
    for j in J:
        if j not in d:
            raise ValueError(f'Missing demand for customer {j}')
    c = {}
    for (_, row) in df_transport.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in I:
            raise ValueError(f'Unexpected supplier {supplier} in transportation_costs.csv')
        c[supplier] = {}
        for j in J:
            c[supplier][j] = float(row[j])
    for i in I:
        if i not in c:
            raise ValueError(f'Missing transportation costs for supplier {i}')
        for j in J:
            if j not in c[i]:
                raise ValueError(f'Missing transportation cost for supplier {i} to customer {j}')
    M = sum((d[j] for j in J))
    m = gurobipy.Model('UFLP5')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((f[i] * y[i] for i in I)) + gurobipy.quicksum((c[i][j] * x[i, j] for i in I for j in J))
    m.setObjective(obj, GRB.MINIMIZE)
    for j in J:
        m.addConstr(gurobipy.quicksum((x[i, j] for i in I)) == d[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gurobipy.quicksum((x[i, j] for j in J)) <= M * y[i], name=f'activation_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')