import gurobipy as gp
import pandas as pd
import numpy as np

def solve_generation_lot_sizing():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv', dtype=str, keep_default_na=False)
    if 'option' not in df.columns:
        raise KeyError("Missing required column 'option' in energy.csv")
    option_ids = df['option'].tolist()
    if len(option_ids) != len(set(option_ids)):
        raise ValueError("Duplicate option identifiers found in 'option' column.")
    required_cols = ['gen_per_lot', 'cost_per_lot']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in energy.csv")
    try:
        gen_per_lot = df.set_index('option')['gen_per_lot'].astype(int).to_dict()
        cost_per_lot = df.set_index('option')['cost_per_lot'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting parameter columns to numeric: {e}')
    for i in option_ids:
        if i not in gen_per_lot or i not in cost_per_lot:
            raise ValueError(f"Missing coefficients for option '{i}'.")
    total_demand = 200
    m = gp.Model('generation_lot_sizing')
    x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x_vars[i] for i in option_ids)) >= total_demand, name='demand')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in option_ids:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_generation_lot_sizing()