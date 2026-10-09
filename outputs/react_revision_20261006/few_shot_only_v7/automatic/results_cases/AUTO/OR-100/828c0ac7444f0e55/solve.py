import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'processing_time_unit': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv', 'unit_price': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv', 'total_working_hours': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_fallback(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except Exception:
                continue
        raise RuntimeError(f'Failed to read {path} with tried encodings.')
    df_time = read_csv_with_fallback(csv_paths['processing_time_unit'])
    df_price = read_csv_with_fallback(csv_paths['unit_price'])
    df_hours = read_csv_with_fallback(csv_paths['total_working_hours'])
    workshops = list(df_hours['workshop'])
    component_cols = [col for col in df_time.columns if col != 'Unnamed: 0']
    components = component_cols
    processing_time = {}
    for (w_idx, w) in enumerate(df_time['Unnamed: 0']):
        for c in components:
            val = df_time.at[w_idx, c]
            if val == '':
                raise ValueError(f'Missing processing time for workshop {w}, component {c}')
            try:
                processing_time[w, c] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric processing time for workshop {w}, component {c}: {val}')
    if len(df_price) != len(components):
        raise ValueError('unit_price.csv row count does not match number of components')
    unit_price = {}
    for (idx, c) in enumerate(components):
        val = df_price.at[idx, 'unit_price']
        if val == '':
            raise ValueError(f'Missing unit price for component {c}')
        try:
            unit_price[c] = float(val)
        except Exception:
            raise ValueError(f'Non-numeric unit price for component {c}: {val}')
    total_hours = {}
    for (idx, row) in df_hours.iterrows():
        w = row['workshop']
        val = row['total_hours']
        if val == '':
            raise ValueError(f'Missing total_hours for workshop {w}')
        try:
            total_hours[w] = float(val)
        except Exception:
            raise ValueError(f'Non-numeric total_hours for workshop {w}: {val}')
    for w in workshops:
        for c in components:
            if (w, c) not in processing_time:
                raise ValueError(f'Missing processing time for workshop {w}, component {c}')
    for c in components:
        if c not in unit_price:
            raise ValueError(f'Missing unit price for component {c}')
    for w in workshops:
        if w not in total_hours:
            raise ValueError(f'Missing total_hours for workshop {w}')
    m = gp.Model('factory_production')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(components, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((unit_price[c] * quantity_vars[c] for c in components)), GRB.MAXIMIZE)
    for w in workshops:
        m.addConstr(gp.quicksum((processing_time[w, c] * quantity_vars[c] for c in components)) <= total_hours[w], name=f'cap_{w}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')