LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv", "values": {"id_number": "id999", "Revenue": "434.74", "Demand": "8171", "Initial Inventory": "56450"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in records:
    vals = rec['values']
    if vals.get('id_number') == 'id999':
        pid = vals['id_number']
        products.append(pid)
        try:
            revenue[pid] = float(vals['Revenue'])
            demand[pid] = int(vals['Demand'])
            initial_inventory[pid] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pid}: {e}')
for pid in products:
    if pid not in revenue or pid not in demand or pid not in initial_inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('Supermarket_Fulfillment')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= initial_inventory[i] for i in products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')