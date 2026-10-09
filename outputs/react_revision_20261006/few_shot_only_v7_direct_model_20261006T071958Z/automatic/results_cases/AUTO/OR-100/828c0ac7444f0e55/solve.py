import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'processing_time_unit': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv', 'unit_price': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv', 'total_working_hours': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    df_ptu = read_csv_with_encodings(csv_paths['processing_time_unit'])
    df_up = read_csv_with_encodings(csv_paths['unit_price'])
    df_twh = read_csv_with_encodings(csv_paths['total_working_hours'])
    C = list(df_up['Unnamed: 0'])
    W = list(df_ptu['Unnamed: 0'])
    ptu_cols = [col for col in df_ptu.columns if col != 'Unnamed: 0']
    if set(C) != set(ptu_cols):
        raise ValueError('Component set C from unit_price.csv does not match columns in processing_time_unit.csv.')
    twh_ws = list(df_twh['workshop'])
    if set(W) != set(twh_ws):
        raise ValueError('Workshop set W from processing_time_unit.csv does not match workshops in total_working_hours.csv.')
    try:
        p_c = {row['Unnamed: 0']: float(row['unit_price']) for (_, row) in df_up.iterrows()}
    except Exception as e:
        raise ValueError(f'Error converting unit_price to float: {e}')
    a_wc = {}
    for (_, row) in df_ptu.iterrows():
        w = row['Unnamed: 0']
        for c in C:
            try:
                a_wc[w, c] = float(row[c])
            except Exception as e:
                raise ValueError(f'Error converting processing_time_unit for ({w},{c}) to float: {e}')
    try:
        b_w = {row['workshop']: float(row['total_hours']) for (_, row) in df_twh.iterrows()}
    except Exception as e:
        raise ValueError(f'Error converting total_hours to float: {e}')
    m = gp.Model('component_production')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(C, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((p_c[c] * quantity_vars[c] for c in C)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((a_wc[w, c] * quantity_vars[c] for c in C)) <= b_w[w] for w in W), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()