LEGACY_OBSERVATION = 'Product Name,Revenue,Demand,Initial Inventory\nTABLET_10084.74,10084.74,2,10\nTABLET_12211.86,12211.86,43,300\nTABLET_14669.5,14669.5,6,30\nTABLET_14745.76,14745.76,6,30\nTABLET_14754.24,14754.24,20,100\nTABLET_16448.3,16448.3,22,110\nTABLET_16448.31,16448.31,3,20\nTABLET_20143.22,20143.22,16,80\nTABLET_2042.38,2042.38,2,10\nTABLET_24915.25,24915.25,2,10\nTABLET_24915.26,24915.26,32,160\nTABLET_26448.3,26448.3,14,70\nTABLET_27042.38,27042.38,2,10\nTABLET_30000.0,30000.0,2,10\nTABLET_33397.46,33397.46,6,30\nTABLET_33398.3,33398.3,2,10\nTABLET_48567.8,48567.8,6,30\nTABLET_48644.07,48644.07,2,10\nTABLET_50262.72,50262.72,6,30\nTABLET_53736.44,53736.44,6,30\nTABLET_6957.62,6957.62,6,40\nTABLET_6957.63,6957.63,12,80\nTABLET_7550.84,7550.84,60,300\nTABLET_7550.85,7550.85,8,40\nTABLET_9584.74,9584.74,8,40\nTABLET_9661.02,9661.02,38,190\nTABLET_9669.5,9669.5,4,20'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'TABLET_10084.74', 'Revenue': '10084.74', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'TABLET_12211.86', 'Revenue': '12211.86', 'Demand': '43', 'Initial Inventory': '300'}}, {'source': '', 'values': {'Product Name': 'TABLET_14669.5', 'Revenue': '14669.5', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'TABLET_14745.76', 'Revenue': '14745.76', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'TABLET_14754.24', 'Revenue': '14754.24', 'Demand': '20', 'Initial Inventory': '100'}}, {'source': '', 'values': {'Product Name': 'TABLET_16448.3', 'Revenue': '16448.3', 'Demand': '22', 'Initial Inventory': '110'}}, {'source': '', 'values': {'Product Name': 'TABLET_16448.31', 'Revenue': '16448.31', 'Demand': '3', 'Initial Inventory': '20'}}, {'source': '', 'values': {'Product Name': 'TABLET_20143.22', 'Revenue': '20143.22', 'Demand': '16', 'Initial Inventory': '80'}}, {'source': '', 'values': {'Product Name': 'TABLET_2042.38', 'Revenue': '2042.38', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'TABLET_24915.25', 'Revenue': '24915.25', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'TABLET_24915.26', 'Revenue': '24915.26', 'Demand': '32', 'Initial Inventory': '160'}}, {'source': '', 'values': {'Product Name': 'TABLET_26448.3', 'Revenue': '26448.3', 'Demand': '14', 'Initial Inventory': '70'}}, {'source': '', 'values': {'Product Name': 'TABLET_27042.38', 'Revenue': '27042.38', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'TABLET_30000.0', 'Revenue': '30000.0', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'TABLET_33397.46', 'Revenue': '33397.46', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'TABLET_33398.3', 'Revenue': '33398.3', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'TABLET_48567.8', 'Revenue': '48567.8', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'TABLET_48644.07', 'Revenue': '48644.07', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'TABLET_50262.72', 'Revenue': '50262.72', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'TABLET_53736.44', 'Revenue': '53736.44', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'TABLET_6957.62', 'Revenue': '6957.62', 'Demand': '6', 'Initial Inventory': '40'}}, {'source': '', 'values': {'Product Name': 'TABLET_6957.63', 'Revenue': '6957.63', 'Demand': '12', 'Initial Inventory': '80'}}, {'source': '', 'values': {'Product Name': 'TABLET_7550.84', 'Revenue': '7550.84', 'Demand': '60', 'Initial Inventory': '300'}}, {'source': '', 'values': {'Product Name': 'TABLET_7550.85', 'Revenue': '7550.85', 'Demand': '8', 'Initial Inventory': '40'}}, {'source': '', 'values': {'Product Name': 'TABLET_9584.74', 'Revenue': '9584.74', 'Demand': '8', 'Initial Inventory': '40'}}, {'source': '', 'values': {'Product Name': 'TABLET_9661.02', 'Revenue': '9661.02', 'Demand': '38', 'Initial Inventory': '190'}}, {'source': '', 'values': {'Product Name': 'TABLET_9669.5', 'Revenue': '9669.5', 'Demand': '4', 'Initial Inventory': '20'}}]
import gurobipy as gp
from gurobipy import GRB
tablets = []
revenue = {}
demand = {}
init_inventory = {}
capacity = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    if pname.startswith('TABLET_'):
        tablets.append(pname)
        try:
            revenue[pname] = float(vals['Revenue'])
            demand[pname] = int(vals['Demand'])
            init_inventory[pname] = int(vals['Initial Inventory'])
            capacity[pname] = min(demand[pname], init_inventory[pname])
        except Exception as e:
            raise ValueError(f'Invalid data for {pname}: {e}')
for pname in tablets:
    if pname not in revenue or pname not in demand or pname not in init_inventory or (pname not in capacity):
        raise ValueError(f'Missing data for {pname}')
m = gp.Model('TABLET_Fulfillment')
x = m.addVars(tablets, lb=0, ub=[capacity[i] for i in tablets], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in tablets)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= capacity[i] for i in tablets), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')