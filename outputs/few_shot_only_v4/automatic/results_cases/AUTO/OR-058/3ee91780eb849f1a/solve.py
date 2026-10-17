import gurobipy as gp
from gurobipy import GRB
import pandas as pd
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', dtype=str, keep_default_na=False)
if demand_df.shape[1] == 2:
    demand_keys = demand_df.iloc[:, 0].tolist()
    demand_vals = [float(x) for x in demand_df.iloc[:, 1]]
    demand = dict(zip(demand_keys, demand_vals))
else:
    raise ValueError('demand.csv must have two columns: store, demand')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', dtype=str, keep_default_na=False)
if fixed_cost_df.shape[1] == 2:
    supplier_keys = fixed_cost_df.iloc[:, 0].tolist()
    fixed_cost_vals = [float(x) for x in fixed_cost_df.iloc[:, 1]]
    fixed_cost = dict(zip(supplier_keys, fixed_cost_vals))
else:
    raise ValueError('fixed_cost.csv must have two columns: supplier, fixed_cost')
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', dtype=str, keep_default_na=False)
if trans_costs_df.shape[1] < 2:
    raise ValueError('transportation_costs.csv must have at least two columns: supplier and at least one store')
trans_costs_df = trans_costs_df.set_index(trans_costs_df.columns[0])
suppliers = trans_costs_df.index.tolist()
stores = trans_costs_df.columns.tolist()
cost = {i: {j: float(trans_costs_df.loc[i, j]) for j in stores} for i in suppliers}
if set(demand.keys()) != set(stores):
    raise ValueError('Mismatch between demand stores and transportation_costs columns')
if set(fixed_cost.keys()) != set(suppliers):
    raise ValueError('Mismatch between fixed_cost suppliers and transportation_costs rows')
M = sum((demand[j] for j in stores))
m = gp.Model('Original_RAG_FLP')
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