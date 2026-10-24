import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path_time = 'Test_Dataset/Large-scale-or/Others_example/Others8/processing_time_unit.csv'
    path_price = 'Test_Dataset/Large-scale-or/Others_example/Others8/unit_price.csv'
    path_hours = 'Test_Dataset/Large-scale-or/Others_example/Others8/total_working_hours.csv'
    df_time = pd.read_csv(path_time, dtype=str, keep_default_na=False, index_col=0)
    df_time = df_time.applymap(lambda x: float(x) if x != '' else 0.0)
    df_price = pd.read_csv(path_price, dtype=str, keep_default_na=False, index_col=0)
    df_price['unit_price'] = df_price['unit_price'].apply(lambda x: float(x) if x != '' else 0.0)
    price_dict = df_price['unit_price'].to_dict()
    df_hours = pd.read_csv(path_hours, dtype=str, keep_default_na=False, index_col=0)
    df_hours['total_hours'] = df_hours['total_hours'].apply(lambda x: float(x) if x != '' else 0.0)
    hours_dict = df_hours['total_hours'].to_dict()
    components = list(price_dict.keys())
    workshops = list(hours_dict.keys())
    for w in workshops:
        if w not in df_time.index:
            raise ValueError(f"Missing processing time data for workshop '{w}'")
    for c in components:
        if c not in df_time.columns:
            raise ValueError(f"Missing processing time data for component '{c}'")
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(components, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((price_dict[i] * x[i] for i in components)), GRB.MAXIMIZE)
    for w in workshops:
        expr = gurobipy.quicksum((df_time.loc[w, i] * x[i] for i in components))
        m.addConstr(expr <= hours_dict[w], name='cap_' + w.replace(' ', '_'))
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()