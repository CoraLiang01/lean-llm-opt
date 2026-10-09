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
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path, dtype=str, keep_default_na=False)
    fixed_cost_df = read_csv_with_encodings(fixed_cost_path, dtype=str, keep_default_na=False)
    trans_cost_df = read_csv_with_encodings(trans_cost_path, dtype=str, keep_default_na=False)
    I_fc = fixed_cost_df['Unnamed: 0'].tolist()
    I_tc = trans_cost_df['Unnamed: 0'].tolist()
    I = sorted(set(I_fc) | set(I_tc))
    J_demand = demand_df['customer'].tolist()
    J_tc = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    J = sorted(set(J_demand) | set(J_tc))
    demand_map = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        if cust not in J:
            continue
        try:
            val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        if cust in demand_map:
            demand_map[cust] += val
        else:
            demand_map[cust] = val
    for j in J:
        if j not in demand_map:
            raise ValueError(f'Missing demand for customer {j}')
    fixed_cost_map = {}
    for (_, row) in fixed_cost_df.iterrows():
        fac = row['Unnamed: 0']
        if fac not in I:
            continue
        try:
            val = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for warehouse {fac}: {row['fixed_costs']}")
        if fac in fixed_cost_map:
            fixed_cost_map[fac] += val
        else:
            fixed_cost_map[fac] = val
    for i in I:
        if i not in fixed_cost_map:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
    cost_map = {}
    for (_, row) in trans_cost_df.iterrows():
        fac = row['Unnamed: 0']
        if fac not in I:
            continue
        cost_map[fac] = {}
        for cust in J:
            if cust in trans_cost_df.columns:
                val = row[cust]
            else:
                val = None
                for col in trans_cost_df.columns:
                    if col == 'Unnamed: 0':
                        continue
                    if col.casefold() == cust.casefold():
                        val = row[col]
                        break
            if val is None or val == '':
                raise ValueError(f'Missing transportation cost for warehouse {fac}, customer {cust}')
            try:
                cost_map[fac][cust] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for warehouse {fac}, customer {cust}: {val}')
    for i in I:
        if i not in cost_map:
            raise ValueError(f'Missing transportation cost row for warehouse {i}')
        for j in J:
            if j not in cost_map[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i}, customer {j}')
    M = sum((demand_map[j] for j in J))
    m = gp.Model('Bandcamp_FLP')
    x_keys = [(i, j) for i in I for j in J]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost_map[i][j] * x_vars[i, j] for (i, j) in x_keys)) + gp.quicksum((fixed_cost_map[i] * y_vars[i] for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in I)) == demand_map[j], name=f'demand_{j}')
    for i in I:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in J)) <= M * y_vars[i], name=f'activation_{i}')
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