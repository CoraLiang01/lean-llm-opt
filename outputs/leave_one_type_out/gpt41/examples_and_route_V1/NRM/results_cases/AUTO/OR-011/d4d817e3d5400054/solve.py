LEGACY_OBSERVATION = 'Product,Revenue,Initial Inventory,Demand\nid999,434.74,56450,8171'
LEGACY_RECORDS = [{'source': '', 'values': {'Product': 'id999', 'Revenue': '434.74', 'Initial Inventory': '56450', 'Demand': '8171'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
initial_inventory = {}
demand = {}
for rec in records:
    vals = rec['values']
    if vals.get('Product') == 'id999':
        pid = vals['Product']
        products.append(pid)
        try:
            revenue[pid] = float(vals['Revenue'])
            initial_inventory[pid] = int(vals['Initial Inventory'])
            demand[pid] = int(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {pid}: {e}')
for pid in products:
    if pid not in revenue or pid not in initial_inventory or pid not in demand:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('id999_fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
for i in products:
    m.addConstr(x[i] <= initial_inventory[i], name=f'inv_{i}')
    m.addConstr(x[i] <= demand[i], name=f'dem_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')