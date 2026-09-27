CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail establishment seeks to optimize the allocation of its merchandise across distinct categories '
          '(electronics, apparel, homeware, etc.) to achieve the highest possible total revenue. Each category '
          'maintains its own demand level, with revenue figures provided in the ‘Revenue’ column and current stock '
          'quantities documented in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The optimization challenge involves '
          'determining the ideal fulfillment quantities x_i for each product that allocate available inventory while '
          'respecting inventory limits and maximizing total revenue.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailSalesDataset.csv',
             'filters': {},
             'original_rows': 35,
             'records': [{'source_row': 0,
                          'values': {'Demand': '240',
                                     'Initial Inventory': '1570',
                                     'Product Name': 'Beauty - 25',
                                     'Revenue': '25'}},
                         {'source_row': 1,
                          'values': {'Demand': '202',
                                     'Initial Inventory': '1330',
                                     'Product Name': 'Beauty - 30',
                                     'Revenue': '30'}},
                         {'source_row': 2,
                          'values': {'Demand': '216',
                                     'Initial Inventory': '1420',
                                     'Product Name': 'Beauty - 300',
                                     'Revenue': '300'}},
                         {'source_row': 3,
                          'values': {'Demand': '263',
                                     'Initial Inventory': '1700',
                                     'Product Name': 'Beauty - 50',
                                     'Revenue': '50'}},
                         {'source_row': 4,
                          'values': {'Demand': '256',
                                     'Initial Inventory': '1690',
                                     'Product Name': 'Beauty - 500',
                                     'Revenue': '500'}},
                         {'source_row': 5,
                          'values': {'Demand': '281',
                                     'Initial Inventory': '1840',
                                     'Product Name': 'Clothing - 25',
                                     'Revenue': '25'}},
                         {'source_row': 6,
                          'values': {'Demand': '261',
                                     'Initial Inventory': '1710',
                                     'Product Name': 'Clothing - 30',
                                     'Revenue': '30'}},
                         {'source_row': 7,
                          'values': {'Demand': '295',
                                     'Initial Inventory': '1930',
                                     'Product Name': 'Clothing - 300',
                                     'Revenue': '300'}},
                         {'source_row': 8,
                          'values': {'Demand': '290',
                                     'Initial Inventory': '1890',
                                     'Product Name': 'Clothing - 50',
                                     'Revenue': '50'}},
                         {'source_row': 9,
                          'values': {'Demand': '244',
                                     'Initial Inventory': '1570',
                                     'Product Name': 'Clothing - 500',
                                     'Revenue': '500'}},
                         {'source_row': 10,
                          'values': {'Demand': '273',
                                     'Initial Inventory': '1810',
                                     'Product Name': 'Electronics - 25',
                                     'Revenue': '25'}},
                         {'source_row': 11,
                          'values': {'Demand': '220',
                                     'Initial Inventory': '1410',
                                     'Product Name': 'Electronics - 30',
                                     'Revenue': '30'}},
                         {'source_row': 12,
                          'values': {'Demand': '286',
                                     'Initial Inventory': '1830',
                                     'Product Name': 'Electronics - 300',
                                     'Revenue': '300'}},
                         {'source_row': 13,
                          'values': {'Demand': '268',
                                     'Initial Inventory': '1750',
                                     'Product Name': 'Electronics - 50',
                                     'Revenue': '50'}},
                         {'source_row': 14,
                          'values': {'Demand': '262',
                                     'Initial Inventory': '1690',
                                     'Product Name': 'Electronics - 500',
                                     'Revenue': '500'}},
                         {'source_row': 15,
                          'values': {'Demand': '255',
                                     'Initial Inventory': '1660',
                                     'Product Name': 'Home Goods - 25',
                                     'Revenue': '25'}},
                         {'source_row': 16,
                          'values': {'Demand': '218',
                                     'Initial Inventory': '1417',
                                     'Product Name': 'Home Goods - 30',
                                     'Revenue': '30'}},
                         {'source_row': 17,
                          'values': {'Demand': '278',
                                     'Initial Inventory': '1807',
                                     'Product Name': 'Home Goods - 50',
                                     'Revenue': '50'}},
                         {'source_row': 18,
                          'values': {'Demand': '195',
                                     'Initial Inventory': '1268',
                                     'Product Name': 'Home Goods - 300',
                                     'Revenue': '300'}},
                         {'source_row': 19,
                          'values': {'Demand': '248',
                                     'Initial Inventory': '1612',
                                     'Product Name': 'Home Goods - 500',
                                     'Revenue': '500'}},
                         {'source_row': 20,
                          'values': {'Demand': '269',
                                     'Initial Inventory': '1749',
                                     'Product Name': 'Sports - 25',
                                     'Revenue': '25'}},
                         {'source_row': 21,
                          'values': {'Demand': '227',
                                     'Initial Inventory': '1476',
                                     'Product Name': 'Sports - 30',
                                     'Revenue': '30'}},
                         {'source_row': 22,
                          'values': {'Demand': '285',
                                     'Initial Inventory': '1853',
                                     'Product Name': 'Sports - 50',
                                     'Revenue': '50'}},
                         {'source_row': 23,
                          'values': {'Demand': '208',
                                     'Initial Inventory': '1352',
                                     'Product Name': 'Sports - 300',
                                     'Revenue': '300'}},
                         {'source_row': 24,
                          'values': {'Demand': '259',
                                     'Initial Inventory': '1684',
                                     'Product Name': 'Sports - 500',
                                     'Revenue': '500'}},
                         {'source_row': 25,
                          'values': {'Demand': '242',
                                     'Initial Inventory': '1573',
                                     'Product Name': 'Furniture - 25',
                                     'Revenue': '25'}},
                         {'source_row': 26,
                          'values': {'Demand': '213',
                                     'Initial Inventory': '1385',
                                     'Product Name': 'Furniture - 30',
                                     'Revenue': '30'}},
                         {'source_row': 27,
                          'values': {'Demand': '272',
                                     'Initial Inventory': '1768',
                                     'Product Name': 'Furniture - 50',
                                     'Revenue': '50'}},
                         {'source_row': 28,
                          'values': {'Demand': '224',
                                     'Initial Inventory': '1456',
                                     'Product Name': 'Furniture - 300',
                                     'Revenue': '300'}},
                         {'source_row': 29,
                          'values': {'Demand': '251',
                                     'Initial Inventory': '1632',
                                     'Product Name': 'Furniture - 500',
                                     'Revenue': '500'}},
                         {'source_row': 30,
                          'values': {'Demand': '257',
                                     'Initial Inventory': '1671',
                                     'Product Name': 'Toys - 25',
                                     'Revenue': '25'}},
                         {'source_row': 31,
                          'values': {'Demand': '235',
                                     'Initial Inventory': '1528',
                                     'Product Name': 'Toys - 30',
                                     'Revenue': '30'}},
                         {'source_row': 32,
                          'values': {'Demand': '291',
                                     'Initial Inventory': '1892',
                                     'Product Name': 'Toys - 50',
                                     'Revenue': '50'}},
                         {'source_row': 33,
                          'values': {'Demand': '199',
                                     'Initial Inventory': '1294',
                                     'Product Name': 'Toys - 300',
                                     'Revenue': '300'}},
                         {'source_row': 34,
                          'values': {'Demand': '264',
                                     'Initial Inventory': '1716',
                                     'Product Name': 'Toys - 500',
                                     'Revenue': '500'}}],
             'returned_rows': 35,
             'role': 'revenue-management objective and constraints',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = table['records']
items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    try:
        rev = float(vals['Revenue'])
        dem = int(vals['Demand'])
        inv = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{prod}': {e}")
    items.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv
if not set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(items):
    raise ValueError('Mismatch in product identifiers among parameters.')
m = gp.Model('Retail_Revenue_Allocation')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in items), name='')
m.addConstrs((x[i] <= demand[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')