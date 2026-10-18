LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv", "values": {"id_number": "id999", "Revenue": "434.74", "Demand": "8171", "Initial Inventory": "56450"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    if vals.get('id_number') == 'id999':
        pid = vals['id_number']
        products.append(pid)
        try:
            revenue[pid] = float(vals['Revenue'])
            demand[pid] = int(vals['Demand'])
            inventory[pid] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Missing or invalid data for product {pid}: {e}')
for pid in products:
    if pid not in revenue or pid not in demand or pid not in inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('RetailRevenueMax')
x_vars = m.addVars(products, lb=0, ub={pid: min(inventory[pid], demand[pid]) for pid in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')