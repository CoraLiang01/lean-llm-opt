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
table_id = 'file_0_view_0'
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == table_id:
        table = t
        break
if table is None:
    raise ValueError(f'Table {table_id} not found in CSVQA_DATA.')
records = table['records']
if not records:
    raise ValueError(f'No records found in table {table_id}.')
col_product = 'Product Name'
col_revenue = 'Revenue'
col_demand = 'Demand'
col_inventory = 'Initial Inventory'
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    try:
        product = vals[col_product]
        if product in products:
            raise ValueError(f'Duplicate product identifier: {product}')
        products.append(product)
        rev_match = re.fullmatch('\\s*([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\s*', vals[col_revenue])
        dem_match = re.fullmatch('\\s*([+-]?\\d+)\\s*', vals[col_demand])
        inv_match = re.fullmatch('\\s*([+-]?\\d+)\\s*', vals[col_inventory])
        if not rev_match:
            raise ValueError(f"Revenue value '{vals[col_revenue]}' for product '{product}' is not a valid number.")
        if not dem_match:
            raise ValueError(f"Demand value '{vals[col_demand]}' for product '{product}' is not a valid integer.")
        if not inv_match:
            raise ValueError(f"Initial Inventory value '{vals[col_inventory]}' for product '{product}' is not a valid integer.")
        revenue[product] = float(rev_match.group(1))
        demand[product] = int(dem_match.group(1))
        inventory[product] = int(inv_match.group(1))
    except KeyError as e:
        raise ValueError(f'Missing required column {e} in record: {rec}')
if not set(products) == set(revenue) == set(demand) == set(inventory):
    raise ValueError('Mismatch in product identifiers among products, revenue, demand, or inventory.')
m = gp.Model('WomenClothingEcommerceSales')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
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