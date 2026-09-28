import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
warehouse_df = pd.read_csv(warehouse_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
warehouse_ids = warehouse_df['Warehouse ID'].astype(str).tolist()
customer_ids = demand_df['Customer ID'].astype(str).tolist()
fixed_cost = warehouse_df.set_index('Warehouse ID')['Fixed_Cost'].astype(float).to_dict()
capacity = warehouse_df.set_index('Warehouse ID')['Capacity'].astype(float).to_dict()
demand = demand_df.set_index('Customer ID')['Demand'].astype(float).to_dict()
cost_matrix = {}
cost_df_indexed = cost_df.set_index('Warehouse ID')
for i in warehouse_ids:
    cost_matrix[i] = {}
    for j in customer_ids:
        if j not in cost_df_indexed.columns:
            raise KeyError(f"Customer ID '{j}' not found as a column in cost.csv")
        cost_matrix[i][j] = float(cost_df_indexed.loc[i, j])
if set(warehouse_ids) != set(cost_df['Warehouse ID'].astype(str)):
    raise ValueError('Mismatch between warehouse IDs in warehouse.csv and cost.csv')
if set(customer_ids) != set([col for col in cost_df.columns if col != 'Warehouse ID']):
    raise ValueError('Mismatch between customer IDs in demand.csv and cost.csv columns')
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(warehouse_ids, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouse_ids, customer_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in warehouse_ids)) + gp.quicksum((cost_matrix[i][j] * x[i, j] for i in warehouse_ids for j in customer_ids)), gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouse_ids)) == demand[j], name=f'demand_{j}')
for i in warehouse_ids:
    m.addConstr(gp.quicksum((x[i, j] for j in customer_ids)) <= capacity[i] * y[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Warehouses Opened ---')
    for i in warehouse_ids:
        if y[i].X > 0.5:
            print(f'  Warehouse {i}: OPEN (Fixed Cost: {fixed_cost[i]:.2f}, Capacity: {capacity[i]:.0f})')
    print('\n--- Customer Assignments ---')
    for j in customer_ids:
        print(f'Customer {j} (Demand: {demand[j]:.0f}):')
        for i in warehouse_ids:
            if x[i, j].X > 1e-06:
                print(f'    Served by Warehouse {i}: {x[i, j].X:.2f} units (Cost per unit: {cost_matrix[i][j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')