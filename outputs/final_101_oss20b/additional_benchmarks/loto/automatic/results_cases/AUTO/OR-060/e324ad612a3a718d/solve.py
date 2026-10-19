import gurobipy as gp
import pandas as pd
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
df_demand = pd.read_csv(demand_path, sep=',')
df_fixed = pd.read_csv(fixed_cost_path, sep=',')
df_transport = pd.read_csv(transport_path, sep=',')
suppliers = df_fixed['Unnamed: 0'].tolist()
customers = df_demand['customer'].tolist()
fixed_cost = dict(zip(df_fixed['Unnamed: 0'], df_fixed['fixed_costs']))
demand = dict(zip(df_demand['customer'], df_demand['demand']))
df_transport.set_index('Unnamed: 0', inplace=True)
transport_cost = {supplier: dict(zip(customers, df_transport.loc[supplier, customers].values)) for supplier in suppliers}
M = df_demand['demand'].sum()
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS)
obj_fixed = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
obj_transport = gp.quicksum((transport_cost[i][j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj_fixed + obj_transport, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) >= demand[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\nSupplier activation:')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i}: OPEN (fixed cost {fixed_cost[i]})')
    print('\nShipment plan:')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.2f} units (transport cost {transport_cost[i][j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')