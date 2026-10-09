import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_try_encodings(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
    transportation_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
    demand_df = read_csv_try_encodings(demand_path, dtype=str, keep_default_na=False)
    if 'Customer' not in demand_df.columns or 'demand' not in demand_df.columns:
        raise ValueError("demand.csv must have columns 'Customer' and 'demand'")
    customers = demand_df['Customer'].tolist()
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['Customer']
        try:
            val = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
        demand[cust] = val
    fixed_cost_df = read_csv_try_encodings(fixed_cost_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in fixed_cost_df.columns or 'fixed_costs' not in fixed_cost_df.columns:
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    suppliers = fixed_cost_df['Unnamed: 0'].tolist()
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        sup = row['Unnamed: 0']
        try:
            val = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for supplier {sup}: {row['fixed_costs']}")
        fixed_cost[sup] = val
    trans_df = read_csv_try_encodings(transportation_costs_path, dtype=str, keep_default_na=False)
    if 'Unnamed: 0' not in trans_df.columns:
        raise ValueError("transportation_costs.csv must have 'Unnamed: 0' as supplier index column")
    cost_customers = [col for col in trans_df.columns if col != 'Unnamed: 0']
    if set(cost_customers) != set(customers):
        raise ValueError(f'Customers in transportation_costs.csv ({cost_customers}) do not match demand.csv ({customers})')
    cost_suppliers = trans_df['Unnamed: 0'].tolist()
    if set(cost_suppliers) != set(suppliers):
        raise ValueError(f'Suppliers in transportation_costs.csv ({cost_suppliers}) do not match fixed_cost.csv ({suppliers})')
    cost = {}
    for (_, row) in trans_df.iterrows():
        sup = row['Unnamed: 0']
        cost[sup] = {}
        for cust in cost_customers:
            try:
                val = float(row[cust])
            except Exception:
                raise ValueError(f'Non-numeric transportation cost for supplier {sup}, customer {cust}: {row[cust]}')
            cost[sup][cust] = val
    I = suppliers
    J = customers
    M = sum((demand[j] for j in J))
    m = gp.Model('Iowa_Liquor_Supplier_Selection')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in I for j in J)) + gp.quicksum((fixed_cost[i] * activation_vars[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) == demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in J)) <= M * activation_vars[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')