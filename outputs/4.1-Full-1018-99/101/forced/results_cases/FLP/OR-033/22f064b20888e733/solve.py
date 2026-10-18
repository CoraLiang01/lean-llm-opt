LEGACY_OBSERVATION = '[{"source": "/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv", "values": {"Product Name": "Baby Food_255.28", "Revenue": "255.28", "Demand": "765850", "Initial Inventory": "5627060"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
baby_products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in records:
    vals = rec['values']
    if 'Product Name' in vals and vals['Product Name'].startswith('Baby'):
        pid = vals['Product Name']
        baby_products.append(pid)
        try:
            revenue[pid] = float(vals['Revenue'])
            demand[pid] = int(vals['Demand'])
            initial_inventory[pid] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pid}: {e}')
for pid in baby_products:
    if pid not in revenue or pid not in demand or pid not in initial_inventory:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('Baby_Product_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(baby_products, lb=0, ub={pid: min(initial_inventory[pid], demand[pid]) for pid in baby_products}, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in baby_products)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')