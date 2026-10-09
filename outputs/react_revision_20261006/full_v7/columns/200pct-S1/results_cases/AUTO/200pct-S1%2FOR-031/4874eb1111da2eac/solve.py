import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = list(energy_df['option'])
for col in ['gen_per_lot', 'cost_per_lot']:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
try:
    gen_per_lot = {}
    cost_per_lot = {}
    for (idx, row) in energy_df.iterrows():
        option = row['option']
        try:
            gen = int(row['gen_per_lot'])
        except Exception:
            raise ValueError(f"Invalid gen_per_lot for option '{option}': {row['gen_per_lot']}")
        try:
            cost = float(row['cost_per_lot'])
        except Exception:
            raise ValueError(f"Invalid cost_per_lot for option '{option}': {row['cost_per_lot']}")
        gen_per_lot[option] = gen
        cost_per_lot[option] = cost
except Exception as e:
    raise RuntimeError(f'Error processing parameter columns: {e}')
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch between option index set and parameter keys.')
demand = 200

def solve_energy_procurement(option_ids, gen_per_lot, cost_per_lot, demand):
    m = gp.Model('energy_procurement')
    quantity_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[opt] * quantity_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[opt] * quantity_vars[opt] for opt in option_ids)) >= demand, name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_energy_procurement(option_ids, gen_per_lot, cost_per_lot, demand)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')