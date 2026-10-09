LEGACY_OBSERVATION = '{"values": {"id_number": "id999", "Revenue": "434.74", "Demand": "8171", "Initial Inventory": "56450"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
import gurobipy as gp
from gurobipy import GRB
records = [{'source': '', 'values': {'id_number': 'id999', 'Revenue': '434.74', 'Demand': '8171', 'Initial Inventory': '56450'}}]
id_numbers = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in records:
    vals = rec['values']
    if 'id_number' not in vals or 'Revenue' not in vals or 'Demand' not in vals or ('Initial Inventory' not in vals):
        raise ValueError('Missing required fields in LEGACY_RECORDS')
    i = vals['id_number']
    id_numbers.append(i)
    revenue[i] = float(vals['Revenue'])
    demand[i] = int(vals['Demand'])
    initial_inventory[i] = int(vals['Initial Inventory'])
m = gp.Model('id999_fulfillment')
x = m.addVars(id_numbers, lb=0, ub=[demand[i] for i in id_numbers], vtype=GRB.INTEGER, name='')
for i in id_numbers:
    m.addConstr(x[i] <= initial_inventory[i], name=f'inv_{i}')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in id_numbers)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')