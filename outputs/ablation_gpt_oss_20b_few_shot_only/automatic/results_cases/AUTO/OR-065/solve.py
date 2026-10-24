import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
    transportation_costs_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
    df_demand = pd.read_csv(demand_path, dtype=str, keep_default_na=False)
    if 'customer' not in df_demand.columns or 'demand' not in df_demand.columns:
        raise ValueError("Demand CSV must contain 'customer' and 'demand' columns.")
    J = list(df_demand['customer'])
    d = {}
    for (_, row) in df_demand.iterrows():
        cust = row['customer']
        try:
            val = float(row['demand'])
        except ValueError:
            raise ValueError(f'Invalid numeric value for demand of customer {cust}.')
        d[cust] = val
    df_fixed = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in df_fixed.columns or 'fixed_costs' not in df_fixed.columns:
        raise ValueError("Fixed cost CSV must contain 'Unnamed: 0' and 'fixed_costs' columns.")
    I = list(df_fixed['Unnamed: 0'])
    f = {}
    for (_, row) in df_fixed.iterrows():
        wh = row['Unnamed: 0']
        try:
            val = float(row['fixed_costs'])
        except ValueError:
            raise ValueError(f'Invalid numeric value for fixed cost of warehouse {wh}.')
        f[wh] = val
    df_trans = pd.read_csv(transportation_costs_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in df_trans.columns:
        raise ValueError("Transportation costs CSV must contain 'Unnamed: 0' column.")
    df_trans.set_index('Unnamed: 0', inplace=True)
    for col in J:
        if col not in df_trans.columns:
            raise ValueError(f'Transportation costs CSV missing column for customer {col}.')
    c = {}
    for wh in I:
        if wh not in df_trans.index:
            raise ValueError(f'Transportation costs CSV missing row for warehouse {wh}.')
        c[wh] = {}
        for cust in J:
            try:
                val = float(df_trans.at[wh, cust])
            except (KeyError, ValueError):
                raise ValueError(f'Invalid or missing transportation cost for warehouse {wh} to customer {cust}.')
            c[wh][cust] = val
    for wh in I:
        if wh not in f:
            raise ValueError(f'Missing fixed cost for warehouse {wh}.')
    for cust in J:
        if cust not in d:
            raise ValueError(f'Missing demand for customer {cust}.')
    for wh in I:
        for cust in J:
            if cust not in c[wh]:
                raise ValueError(f'Missing transportation cost for warehouse {wh} to customer {cust}.')
    M = sum((d[cust] for cust in J))
    m = gurobipy.Model('UFLP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(I, vtype=GRB.BINARY, name='')
    x = m.addVars([(wh, cust) for wh in I for cust in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    obj = gurobipy.quicksum((c[wh][cust] * x[wh, cust] for wh in I for cust in J)) + gurobipy.quicksum((f[wh] * y[wh] for wh in I))
    m.setObjective(obj, GRB.MINIMIZE)
    for cust in J:
        m.addConstr(gurobipy.quicksum((x[wh, cust] for wh in I)) == d[cust], name=f'demand_{cust}')
    for wh in I:
        m.addConstr(gurobipy.quicksum((x[wh, cust] for cust in J)) <= M * y[wh], name=f'activation_{wh}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()