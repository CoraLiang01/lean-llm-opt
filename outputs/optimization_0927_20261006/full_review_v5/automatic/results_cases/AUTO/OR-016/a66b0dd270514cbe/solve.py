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
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'distinct categories (electronics, apparel, homeware, etc.)',
                                         'operator': 'contains',
                                         'value': 'electronics'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'distinct categories (electronics, apparel, homeware, etc.)',
                                         'operator': 'contains',
                                         'value': 'apparel'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'distinct categories (electronics, apparel, homeware, etc.)',
                                         'operator': 'contains',
                                         'value': 'homeware'}],
                         'logic': 'or'},
             'original_rows': 35,
             'records': [{'source_row': 10,
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
                                     'Revenue': '500'}}],
             'returned_rows': 5,
             'role': 'revenue management product-level data',
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
if not records:
    raise ValueError("No records found in table_id 'file_0_view_0'.")
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    try:
        product = vals['Product Name']
        A_i = int(vals['Revenue'])
        d_i = int(vals['Demand'])
        I_i = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing required column: {e}')
    except Exception as e:
        raise ValueError(f'Error parsing record: {e}')
    products.append(product)
    revenue[product] = A_i
    demand[product] = d_i
    inventory[product] = I_i
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('Retail_Revenue_Allocation')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= inventory[p] for p in products), name='')
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')