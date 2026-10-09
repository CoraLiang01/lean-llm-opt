import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path, dtype=str, keep_default_na=False)
    if 'customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    J = demand_df['customer'].tolist()
    d_j = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            demand_val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        if cust in d_j:
            d_j[cust] += demand_val
        else:
            d_j[cust] = demand_val
    fixed_cost_df = read_csv_robust(fixed_cost_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    I = fixed_cost_df['Unnamed: 0'].tolist()
    f_i = {}
    for (_, row) in fixed_cost_df.iterrows():
        fac = row['Unnamed: 0']
        try:
            fixed_val = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed_costs for supplier {fac}: {row['fixed_costs']}")
        if fac in f_i:
            f_i[fac] += fixed_val
        else:
            f_i[fac] = fixed_val
    cost_df = read_csv_robust(transportation_costs_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in cost_df.columns:
        raise ValueError("transportation_costs.csv must have 'Unnamed: 0' as supplier index")
    for cust in J:
        if cust not in cost_df.columns:
            raise ValueError(f'Customer {cust} not found as column in transportation_costs.csv')
    c_ij = {}
    for (_, row) in cost_df.iterrows():
        fac = row['Unnamed: 0']
        if fac not in I:
            continue
        for cust in J:
            try:
                cij_val = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric cost for supplier {fac}, customer {cust}: {row[cust]}')
            c_ij[fac, cust] = cij_val
    for i in I:
        for j in J:
            if (i, j) not in c_ij:
                raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i, j] * quantity_vars[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * activation_vars[i] for i in I)), GRB.MINIMIZE)
    for j in J:
        m.addConstr(gp.quicksum((quantity_vars[i, j] for i in I)) == d_j[j], name=f'demand_{j}')
    for i in I:
        for j in J:
            m.addConstr(quantity_vars[i, j] <= d_j[j] * activation_vars[i], name=f'activation_{i}_{j}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()