import gurobipy as gp
import pandas as pd
customer_demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
transportation_costs_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
customer_df = pd.read_csv(customer_demand_path, sep=',')
supply_df = pd.read_csv(supply_capacity_path, sep=',')
cost_df = pd.read_csv(transportation_costs_path, sep=',')
stores = supply_df['Unnamed: 0'].tolist()
customers = customer_df['customer'].tolist()
supply_capacity = supply_df.set_index('Unnamed: 0')['supply_capacity'].to_dict()
demand = customer_df.set_index('customer')['demand'].to_dict()
cost_matrix = cost_df.set_index('Unnamed: 0')[customers].to_dict(orient='index')
m = gp.Model('TransportationOptimization')
x = m.addVars(stores, customers, vtype=gp.GRB.CONTINUOUS)
m.setObjective(gp.quicksum((cost_matrix[s][c] * x[s, c] for s in stores for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in stores)) == demand[c])
for s in stores:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nTransportation plan (units):')
    for s in stores:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  Store {s} -> Customer {c}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')