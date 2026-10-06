import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    processing_time_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv'
    unit_price_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv'
    total_hours_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'
    df_time = read_csv_with_encodings(processing_time_path)
    df_price = read_csv_with_encodings(unit_price_path)
    df_hours = read_csv_with_encodings(total_hours_path)
    workshops = list(df_hours['workshop'])
    component_cols = [col for col in df_time.columns if col != 'Unnamed: 0']
    components = component_cols
    if 'Unnamed: 0' in df_price.columns:
        df_price = df_price.set_index('Unnamed: 0')
        if len(df_price) != len(components):
            raise ValueError('unit_price.csv row count does not match number of components.')
        unit_price = dict(zip(components, df_price['unit_price']))
    else:
        if len(df_price) != len(components):
            raise ValueError('unit_price.csv row count does not match number of components.')
        unit_price = dict(zip(components, df_price['unit_price']))
    total_hours = dict(zip(df_hours['workshop'], df_hours['total_hours']))
    if 'Unnamed: 0' in df_time.columns:
        df_time = df_time.set_index('Unnamed: 0')
        if not all((w in df_time.index for w in workshops)):
            raise ValueError('Workshop names in total_working_hours.csv do not match those in processing_time_unit.csv.')
        processing_time = {w: df_time.loc[w][components].to_dict() for w in workshops}
    else:
        if len(df_time) != len(workshops):
            raise ValueError('Row count of processing_time_unit.csv does not match number of workshops.')
        processing_time = {w: df_time.iloc[i][components].to_dict() for (i, w) in enumerate(workshops)}
    for w in workshops:
        for c in components:
            if c not in processing_time[w]:
                raise ValueError(f'Missing processing time for workshop {w}, component {c}.')
    for c in components:
        if c not in unit_price:
            raise ValueError(f'Missing unit price for component {c}.')
    for w in workshops:
        if w not in total_hours:
            raise ValueError(f'Missing total hours for workshop {w}.')
    m = gp.Model('factory_production')
    x = m.addVars(components, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((unit_price[c] * x[c] for c in components)), GRB.MAXIMIZE)
    for w in workshops:
        m.addConstr(gp.quicksum((processing_time[w][c] * x[c] for c in components)) <= total_hours[w], name=f'cap_{w}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()