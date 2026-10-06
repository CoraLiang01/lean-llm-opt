LEGACY_OBSERVATION = 'Product_Name,Revenue,Demand,Initial Inventory\nBooks_15.15,15.15,1980,9920.0\nBooks_30.3,30.3,3024,20160.0\nBooks_45.45,45.45,4536,30000.0\nBooks_60.6,60.6,5601,38360.0\nBooks_75.75,75.75,7567,51450.0'
LEGACY_RECORDS = [{'source': '', 'values': {'Product_Name': 'Books_15.15', 'Revenue': '15.15', 'Demand': '1980', 'Initial Inventory': '9920.0'}}, {'source': '', 'values': {'Product_Name': 'Books_30.3', 'Revenue': '30.3', 'Demand': '3024', 'Initial Inventory': '20160.0'}}, {'source': '', 'values': {'Product_Name': 'Books_45.45', 'Revenue': '45.45', 'Demand': '4536', 'Initial Inventory': '30000.0'}}, {'source': '', 'values': {'Product_Name': 'Books_60.6', 'Revenue': '60.6', 'Demand': '5601', 'Initial Inventory': '38360.0'}}, {'source': '', 'values': {'Product_Name': 'Books_75.75', 'Revenue': '75.75', 'Demand': '7567', 'Initial Inventory': '51450.0'}}]
import gurobipy as gp
from gurobipy import GRB
books = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product_Name']
    books.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(float(vals['Demand']))
        inventory[pname] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for product {pname}: {e}')
for pname in books:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f'Missing data for product {pname}')
upper_bound = {p: min(demand[p], inventory[p]) for p in books}
m = gp.Model('Books_Fulfillment')
x = m.addVars(books, lb=0, ub=upper_bound, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in books)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')