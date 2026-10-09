CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 7,
             'returned_rows': 7,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Aalopuri',
                                     'Revenue': '20',
                                     'Demand': '1483',
                                     'Initial Inventory': '10440.0'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Cold coffee',
                                     'Revenue': '40',
                                     'Demand': '1918',
                                     'Initial Inventory': '13610.0'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Frankie',
                                     'Revenue': '50',
                                     'Demand': '1623',
                                     'Initial Inventory': '11500.0'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Panipuri',
                                     'Revenue': '20',
                                     'Demand': '1720',
                                     'Initial Inventory': '12260.0'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Sandwich',
                                     'Revenue': '60',
                                     'Demand': '1558',
                                     'Initial Inventory': '10970.0'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Sugarcane juice',
                                     'Revenue': '25',
                                     'Demand': '1791',
                                     'Initial Inventory': '12780.0'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Vadapav',
                                     'Revenue': '20',
                                     'Demand': '1426',
                                     'Initial Inventory': '10060.0'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import gurobipy as gp
from gurobipy import GRB
import re
table_id = 'file_0_view_0'
found = False
for t in CSVQA_DATA['tables']:
    if t['table_id'] == table_id:
        records = t['records']
        columns = t['columns']
        found = True
        break
if not found:
    raise ValueError(f'Table {table_id} not found in CSVQA_DATA.')
try:
    idx_product = columns.index('Product Name')
    idx_revenue = columns.index('Revenue')
    idx_demand = columns.index('Demand')
    idx_inventory = columns.index('Initial Inventory')
except ValueError as e:
    raise ValueError(f'Required column missing: {e}')
aalop_items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    pname = rec['values']['Product Name']
    if re.search('\\bAalop\\b', pname, re.IGNORECASE):
        aalop_items.append(pname)
        try:
            rev = float(rec['values']['Revenue'])
            dem = int(float(rec['values']['Demand']))
            inv = int(float(rec['values']['Initial Inventory']))
        except Exception as e:
            raise ValueError(f"Failed to parse numeric fields for product '{pname}': {e}")
        revenue[pname] = rev
        demand[pname] = dem
        inventory[pname] = inv
if not aalop_items:
    raise ValueError("No products classified under 'Aalop' found in the data.")
for i in aalop_items:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f"Missing data for product '{i}'.")
m = gp.Model('Aalop_Revenue_Max')
x = m.addVars(aalop_items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in aalop_items)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in aalop_items), name='')
m.addConstrs((x[i] <= demand[i] for i in aalop_items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for variable in m.getVars():
        print(f'{variable.VarName}: {variable.X}')
else:
    print(f'Solver status: {m.Status}')