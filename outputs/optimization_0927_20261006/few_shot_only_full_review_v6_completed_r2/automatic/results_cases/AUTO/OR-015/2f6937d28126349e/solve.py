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
if 'CSVQA_DATA' not in globals():
    raise RuntimeError('CSVQA_DATA is not defined at execution.')
tables = CSVQA_DATA.get('tables', [])
table = next((t for t in tables if t.get('table_id') == table_id), None)
if table is None:
    raise RuntimeError(f'Table {table_id} not found in CSVQA_DATA.')
records = table.get('records', [])
if not records:
    raise RuntimeError(f'No records found in table {table_id}.')
aalop_items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec.get('values', {})
    pname = vals.get('Product Name', '')
    if not isinstance(pname, str):
        continue
    if 'Aalop' in pname:
        aalop_items.append(pname)
        rev_str = str(vals.get('Revenue', '')).strip()
        dem_str = str(vals.get('Demand', '')).strip()
        inv_str = str(vals.get('Initial Inventory', '')).strip()
        rev_match = re.fullmatch('-?\\d+(?:\\.\\d+)?', rev_str)
        dem_match = re.fullmatch('-?\\d+', dem_str)
        inv_match = re.fullmatch('-?\\d+', inv_str)
        if not (rev_match and dem_match and inv_match):
            raise ValueError(f"Unmatched numeric field for product '{pname}': Revenue='{rev_str}', Demand='{dem_str}', Initial Inventory='{inv_str}'")
        revenue[pname] = float(rev_str)
        demand[pname] = int(dem_str)
        inventory[pname] = int(inv_str)
if not aalop_items:
    raise ValueError("No products classified under 'Aalop' found in the data.")
for pname in aalop_items:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing parameter(s) for product '{pname}'.")
m = gp.Model('Aalop_Revenue_Maximization')
x_vars = m.addVars(aalop_items, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in aalop_items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in aalop_items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in aalop_items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')