import gurobipy as gp
import pandas as pd
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
df = pd.read_csv(path, sep=',')
mask = df['id_number'].astype(str).str.startswith('id999')
df_sel = df[mask].copy()
ids = df_sel['id_number'].tolist()
revenue = df_sel.set_index('id_number')['Revenue'].to_dict()
demand = df_sel.set_index('id_number')['Demand'].to_dict()
initial_inventory = df_sel.set_index('id_number')['Initial Inventory'].to_dict()
m = gp.Model('MaxRevenue')
x_vars = {}
for pid in ids:
    var = m.addVar(vtype=gp.GRB.INTEGER)
    x_vars[pid] = var
for pid in ids:
    m.addConstr(x_vars[pid] <= initial_inventory[pid])
    m.addConstr(x_vars[pid] <= demand[pid])
obj = gp.quicksum((revenue[pid] * x_vars[pid] for pid in ids))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Total Revenue: {m.objVal:.2f}')
    print('Fulfilled Quantities:')
    for pid in ids:
        qty = int(x_vars[pid].X)
        if qty > 0:
            print(f'{pid}: {qty}')
else:
    print(f'No optimal solution found. Status: {m.status}')