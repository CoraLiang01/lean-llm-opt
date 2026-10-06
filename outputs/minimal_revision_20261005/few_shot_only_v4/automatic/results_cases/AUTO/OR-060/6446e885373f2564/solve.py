import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
    demand_df = read_csv(demand_path)
    if not {'customer', 'demand'}.issubset(demand_df.columns):
        raise ValueError('demand.csv missing required columns.')
    demand_df['customer'] = demand_df['customer'].astype(str)
    J = demand_df['customer'].tolist()
    d_j = dict(zip(demand_df['customer'], demand_df['demand']))
    fixed_df = read_csv(fixed_cost_path)
    if not {'Unnamed: 0', 'fixed_costs'}.issubset(fixed_df.columns):
        raise ValueError('fixed_cost.csv missing required columns.')
    fixed_df['Unnamed: 0'] = fixed_df['Unnamed: 0'].astype(str)
    I = fixed_df['Unnamed: 0'].tolist()
    f_i = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
    trans_df = read_csv(trans_cost_path)
    if not {'Unnamed: 0'}.union(set(J)).issubset(trans_df.columns):
        raise ValueError('transportation_costs.csv missing required columns.')
    trans_df['Unnamed: 0'] = trans_df['Unnamed: 0'].astype(str)
    missing_suppliers = set(I) - set(trans_df['Unnamed: 0'])
    missing_customers = set(J) - set(trans_df.columns)
    if missing_suppliers:
        raise ValueError(f'transportation_costs.csv missing suppliers: {missing_suppliers}')
    if missing_customers:
        raise ValueError(f'transportation_costs.csv missing supermarkets: {missing_customers}')
    c_ij = {}
    for (_, row) in trans_df.iterrows():
        i = row['Unnamed: 0']
        c_ij[i] = {}
        for j in J:
            c_ij[i][j] = row[j]
    M = sum((d_j[j] for j in J))
    for i in I:
        if i not in f_i:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in c_ij:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in J:
            if j not in c_ij[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, supermarket {j}')
    for j in J:
        if j not in d_j:
            raise ValueError(f'Missing demand for supermarket {j}')
    m = gp.Model('UFLP3')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * x[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * y[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= M * y[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()