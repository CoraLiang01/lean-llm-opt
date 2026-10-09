CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for '
          'these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant '
          'aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, '
          'which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, '
          'and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to '
          'be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The '
          'variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}},
                         {'source_row': 1,
                          'values': {'Demand': '1918',
                                     'Initial Inventory': '13610.0',
                                     'Product Name': 'Cold coffee',
                                     'Revenue': '40'}},
                         {'source_row': 2,
                          'values': {'Demand': '1623',
                                     'Initial Inventory': '11500.0',
                                     'Product Name': 'Frankie',
                                     'Revenue': '50'}},
                         {'source_row': 3,
                          'values': {'Demand': '1720',
                                     'Initial Inventory': '12260.0',
                                     'Product Name': 'Panipuri',
                                     'Revenue': '20'}},
                         {'source_row': 4,
                          'values': {'Demand': '1558',
                                     'Initial Inventory': '10970.0',
                                     'Product Name': 'Sandwich',
                                     'Revenue': '60'}},
                         {'source_row': 5,
                          'values': {'Demand': '1791',
                                     'Initial Inventory': '12780.0',
                                     'Product Name': 'Sugarcane juice',
                                     'Revenue': '25'}},
                         {'source_row': 6,
                          'values': {'Demand': '1426',
                                     'Initial Inventory': '10060.0',
                                     'Product Name': 'Vadapav',
                                     'Revenue': '20'}}],
             'returned_rows': 7,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'fallback_reason': "Filter lacks query evidence: {'column': 'Product Name', 'operator': 'prefix', "
                                   '\'dtype\': \'string\', \'value\': \'Aalop\', \'evidence\': "\'Aalop\' (query), '
                                   'prefix=1 in profile", \'inclusive\': \'both\'}',
                'planner_errors': ["Filter lacks query evidence: {'column': 'Product Name', 'operator': 'prefix', "
                                   '\'dtype\': \'string\', \'value\': \'Aalop\', \'evidence\': "\'Aalop\' (query), '
                                   'prefix=1 in profile", \'inclusive\': \'both\'}'],
                'status': 'FALLBACK_FULL_DATA'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Table with table_id 'file_0_view_0' not found.")
records = table['records']
import re
aalop_pattern = re.compile('^Aalop')
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    name = rec['values']['Product Name']
    if aalop_pattern.match(name):
        products.append(name)
        try:
            revenue[name] = float(rec['values']['Revenue'])
            demand[name] = int(rec['values']['Demand'])
            inventory[name] = float(rec['values']['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Invalid data for product '{name}': {e}")
if not products:
    raise ValueError("No products found with prefix 'Aalop' in 'Product Name'.")
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f"Missing data for product '{i}'.")
m = gp.Model('Aalop_Inventory_Optimization')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')