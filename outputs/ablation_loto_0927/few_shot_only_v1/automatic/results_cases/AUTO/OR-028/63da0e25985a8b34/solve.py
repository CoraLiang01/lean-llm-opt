CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'WomenClothingEcommerceSalesData.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 24,
             'returned_rows': 24,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'sku_I27',
                                     'Revenue': '238',
                                     'Demand': '6',
                                     'Initial Inventory': '30'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'sku_I499',
                                     'Revenue': '287',
                                     'Demand': '4',
                                     'Initial Inventory': '20'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'sku_I719',
                                     'Revenue': '268',
                                     'Demand': '16',
                                     'Initial Inventory': '80'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'sku_T18',
                                     'Revenue': '318',
                                     'Demand': '14',
                                     'Initial Inventory': '70'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'sku_T29',
                                     'Revenue': '207',
                                     'Demand': '4',
                                     'Initial Inventory': '20'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'sku_T39',
                                     'Revenue': '258',
                                     'Demand': '32',
                                     'Initial Inventory': '160'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'sku_T499',
                                     'Revenue': '249',
                                     'Demand': '8',
                                     'Initial Inventory': '40'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'sku_T9',
                                     'Revenue': '227',
                                     'Demand': '2',
                                     'Initial Inventory': '10'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'sku_3081',
                                     'Revenue': '198',
                                     'Demand': '10',
                                     'Initial Inventory': '50'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'sku_339',
                                     'Revenue': '254',
                                     'Demand': '8',
                                     'Initial Inventory': '40'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'sku_3799',
                                     'Revenue': '246',
                                     'Demand': '18',
                                     'Initial Inventory': '90'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'sku_439',
                                     'Revenue': '258',
                                     'Demand': '2',
                                     'Initial Inventory': '10'}},
                         {'source_row': 12,
                          'values': {'Product Name': 'sku_539',
                                     'Revenue': '268',
                                     'Demand': '4',
                                     'Initial Inventory': '20'}},
                         {'source_row': 13,
                          'values': {'Product Name': 'sku_61399',
                                     'Revenue': '278',
                                     'Demand': '8',
                                     'Initial Inventory': '40'}},
                         {'source_row': 14,
                          'values': {'Product Name': 'sku_628',
                                     'Revenue': '268',
                                     'Demand': '2',
                                     'Initial Inventory': '10'}},
                         {'source_row': 15,
                          'values': {'Product Name': 'sku_708',
                                     'Revenue': '298',
                                     'Demand': '198',
                                     'Initial Inventory': '990'}},
                         {'source_row': 16,
                          'values': {'Product Name': 'sku_77',
                                     'Revenue': '258',
                                     'Demand': '32',
                                     'Initial Inventory': '160'}},
                         {'source_row': 17,
                          'values': {'Product Name': 'sku_79',
                                     'Revenue': '315',
                                     'Demand': '18',
                                     'Initial Inventory': '90'}},
                         {'source_row': 18,
                          'values': {'Product Name': 'sku_799',
                                     'Revenue': '264',
                                     'Demand': '570',
                                     'Initial Inventory': '2870'}},
                         {'source_row': 19,
                          'values': {'Product Name': 'sku_8499',
                                     'Revenue': '238',
                                     'Demand': '6',
                                     'Initial Inventory': '30'}},
                         {'source_row': 20,
                          'values': {'Product Name': 'sku_89',
                                     'Revenue': '258',
                                     'Demand': '26',
                                     'Initial Inventory': '130'}},
                         {'source_row': 21,
                          'values': {'Product Name': 'sku_897',
                                     'Revenue': '268',
                                     'Demand': '6',
                                     'Initial Inventory': '30'}},
                         {'source_row': 22,
                          'values': {'Product Name': 'sku_9699',
                                     'Revenue': '288',
                                     'Demand': '33',
                                     'Initial Inventory': '170'}},
                         {'source_row': 23,
                          'values': {'Product Name': 'sku_bobo',
                                     'Revenue': '228',
                                     'Demand': '33',
                                     'Initial Inventory': '170'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import gurobipy as gp
from gurobipy import GRB
import re
if 'CSVQA_DATA' not in globals():
    raise RuntimeError('CSVQA_DATA not found. Please provide CSVQA_DATA at execution.')
tables = CSVQA_DATA.get('tables', [])
table = None
for t in tables:
    if t.get('table_id') == 'file_0_view_0':
        table = t
        break
if table is None:
    raise RuntimeError("Table with table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table.get('records', [])
if not records:
    raise RuntimeError("No records found in table 'file_0_view_0'.")
product_names = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    values = rec.get('values', {})
    for col in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']:
        if col not in values:
            raise RuntimeError(f"Missing column '{col}' in record.")
    pname = values['Product Name']
    rev_match = re.match('^\\s*(-?\\d+(?:\\.\\d+)?)\\s*$', str(values['Revenue']))
    dem_match = re.match('^\\s*(\\d+)\\s*$', str(values['Demand']))
    inv_match = re.match('^\\s*(\\d+)\\s*$', str(values['Initial Inventory']))
    if not rev_match:
        raise RuntimeError(f"Revenue value '{values['Revenue']}' for product '{pname}' is not a valid number.")
    if not dem_match:
        raise RuntimeError(f"Demand value '{values['Demand']}' for product '{pname}' is not a valid non-negative integer.")
    if not inv_match:
        raise RuntimeError(f"Initial Inventory value '{values['Initial Inventory']}' for product '{pname}' is not a valid non-negative integer.")
    product_names.append(pname)
    revenue[pname] = float(rev_match.group(1))
    demand[pname] = int(dem_match.group(1))
    inventory[pname] = int(inv_match.group(1))
if not set(product_names) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
    raise RuntimeError('Mismatch in product identifiers among data columns.')
m = gp.Model('Product_Fulfillment')
x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.addConstrs((x[i] <= demand[i] for i in product_names), name='')
m.addConstrs((x[i] <= inventory[i] for i in product_names), name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')