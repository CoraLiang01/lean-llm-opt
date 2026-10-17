import gurobipy as gp
from gurobipy import GRB
import pandas as pd
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv', dtype=str, keep_default_na=False)
if demand_df.shape[1] == 2:
    demand_keys = demand_df.iloc[:, 0].tolist()
    demand_vals = demand_df.iloc[:, 1].astype(float).tolist()
    demand = dict(zip(demand_keys, demand_vals))
else:
    raise ValueError('demand.csv must have two columns: store/customer and demand.')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv', dtype=str, keep_default_na=False)
if fixed_cost_df.shape[1] == 2:
    supplier_keys = fixed_cost_df.iloc[:, 0].tolist()
    fixed_cost_vals = fixed_cost_df.iloc[:, 1].astype(float).tolist()
    fixed_cost = dict(zip(supplier_keys, fixed_cost_vals))
else:
    raise ValueError('fixed_cost.csv must have two columns: supplier and fixed_cost.')
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv', dtype=str, keep_default_na=False)
if trans_costs_df.shape[1] < 2:
    raise ValueError('transportation_costs.csv must have at least two columns: supplier and at least one store.')
trans_supplier_keys = trans_costs_df.iloc[:, 0].tolist()
trans_store_keys = list(trans_costs_df.columns[1:])
if set(trans_store_keys) != set(demand.keys()):
    raise ValueError('Store/customer names in transportation_costs.csv do not match those in demand.csv.')
cost = {}
for idx, row in trans_costs_df.iterrows():
    supplier = row.iloc[0]
    if supplier not in supplier_keys:
        raise ValueError(f'Supplier {supplier} in transportation_costs.csv not found in fixed_cost.csv.')
    cost[supplier] = {}
    for store in trans_store_keys:
        try:
            cost[supplier][store] = float(row[store])
        except Exception:
            raise ValueError(f'Invalid cost for supplier {supplier}, store {store} in transportation_costs.csv.')
if set(supplier_keys) != set(cost.keys()):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv.')
if set(demand.keys()) != set(trans_store_keys):
    raise ValueError('Mismatch between stores in demand.csv and transportation_costs.csv.')
suppliers = supplier_keys
stores = trans_store_keys
M = sum((demand[j] for j in stores))
m = gp.Model('Iowa_Liquor_UFLP')
x_vars = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= M * y_vars[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')