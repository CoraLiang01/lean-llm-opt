import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_encodings(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    path_unit_price = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv'
    path_processing_time = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv'
    path_total_hours = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'
    df_price = read_csv_encodings(path_unit_price)
    df_time = read_csv_encodings(path_processing_time)
    df_hours = read_csv_encodings(path_total_hours)
    I = df_price['Unnamed: 0'].astype(str).tolist()
    W = df_time['Unnamed: 0'].astype(str).tolist()
    time_cols = [c for c in df_time.columns if c != 'Unnamed: 0']
    if set(I) != set(time_cols):
        raise ValueError('Mismatch between component IDs in unit_price.csv and processing_time_unit.csv columns.')
    p_i = dict(zip(df_price['Unnamed: 0'].astype(str), df_price['unit_price']))
    a_wi = {}
    for (idx, row) in df_time.iterrows():
        w = str(row['Unnamed: 0'])
        for i in I:
            a_wi[w, i] = row[i]
    if 'workshop' in df_hours.columns:
        w_col = 'workshop'
    elif 'Unnamed: 0' in df_hours.columns:
        w_col = 'Unnamed: 0'
    else:
        raise ValueError('Workshop column not found in total_working_hours.csv')
    b_w = dict(zip(df_hours[w_col].astype(str), df_hours['total_hours']))
    if set(W) != set(b_w.keys()):
        raise ValueError('Mismatch between workshops in processing_time_unit.csv and total_working_hours.csv.')
    m = gp.Model('factory_production')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((p_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for w in W:
        m.addConstr(gp.quicksum((a_wi[w, i] * x[i] for i in I)) <= b_w[w], name=f'cap_{w}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()