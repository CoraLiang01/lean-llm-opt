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
                                         'evidence': 'electronics',
                                         'operator': 'prefix',
                                         'value': 'Electronics'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'apparel',
                                         'operator': 'prefix',
                                         'value': 'Apparel'},
                                        {'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'homeware',
                                         'operator': 'prefix',
                                         'value': 'Homeware'}],
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
             'role': 'revenue management product data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = CSVQA_DATA['tables'][0]
records = table['records']
I = []
A = {}
d = {}
I_inv = {}
for rec in records:
    vals = rec['values']
    prod = vals['Product Name']
    try:
        revenue = int(vals['Revenue'])
        demand = int(vals['Demand'])
        inventory = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Non-integer data for product {prod}: {e}')
    I.append(prod)
    A[prod] = revenue
    d[prod] = demand
    I_inv[prod] = inventory
if not set(A.keys()) == set(d.keys()) == set(I_inv.keys()) == set(I):
    raise ValueError('Parameter keys do not match index set I.')
m = gp.Model('Retail_Allocation_Optimization')
x_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((A[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= I_inv[i] for i in I), name='')
m.addConstrs((x_vars[i] <= d[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')