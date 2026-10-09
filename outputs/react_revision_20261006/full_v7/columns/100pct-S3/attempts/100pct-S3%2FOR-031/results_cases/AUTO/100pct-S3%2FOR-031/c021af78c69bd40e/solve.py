import gurobipy as gp
import pandas as pd
import numpy as np

def solve_generation_procurement():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv', sep=',', dtype=str, keep_default_na=False)
    if df['option'].duplicated().any():
        raise ValueError('Duplicate option identifiers found in energy.csv.')
    option_keys = df['option'].tolist()
    try:
        gen_per_lot = df.set_index('option')['gen_per_lot'].astype(float).to_dict()
        cost_per_lot = df.set_index('option')['cost_per_lot'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting gen_per_lot or cost_per_lot to float: {e}')
    for k in option_keys:
        if k not in gen_per_lot or k not in cost_per_lot:
            raise ValueError(f'Missing gen_per_lot or cost_per_lot for option {k}')
    total_demand = 200.0
    m = gp.Model('generation_procurement')
    lot_vars = m.addVars(option_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[k] * lot_vars[k] for k in option_keys)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[k] * lot_vars[k] for k in option_keys)) >= total_demand, name='demand')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for k in option_keys:
            v = lot_vars[k]
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_generation_procurement()