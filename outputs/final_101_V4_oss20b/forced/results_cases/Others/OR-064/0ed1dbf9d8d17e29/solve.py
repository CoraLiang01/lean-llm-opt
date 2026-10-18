import gurobipy as gp
import pandas as pd
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
suppliers = list(fixed_cost_df['Unnamed: 0'])
customers = list(demand_df['customer'])
fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
demand = dict(zip(demand_df['customer'], demand_df['demand']))
transport_df.set_index('Unnamed: 0', inplace=True)
trans_cost = {}
for s in suppliers:
    for c in customers:
        trans_cost[s, c] = transport_df.at[s, c]
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=gp.GRB.CONTINUOUS, lb=0.0)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i])
obj = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((trans_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Open suppliers:')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i} (fixed cost: {fixed_cost[i]:.2f})')
    print('\nShipment plan:')
    for j in customers:
        shipped = sum((x[i, j].X for i in suppliers))
        print(f'  Customer {j}: demand {demand[j]}, shipped {shipped:.2f}')
        for i in suppliers:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'    from {i}: {qty:.2f} units (transport cost {trans_cost[i, j]:.2f})')
else:
    print(f'Optimization ended with status {m.status}')