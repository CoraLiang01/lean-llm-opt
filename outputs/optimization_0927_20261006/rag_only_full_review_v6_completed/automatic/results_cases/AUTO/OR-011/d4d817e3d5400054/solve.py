CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the '
          '‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective '
          'is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. '
          'The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. '
          'During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n'
          '\n'
          'Demand for each product during the sales period is assumed to be deterministic and known in advance, with '
          'demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of '
          'units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative '
          'integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the '
          'fulfillment quantities must satisfy both inventory and demand constraints.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': '‘id999’',
                                         'operator': 'exact',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum
data_table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        data_table = t
        break
if data_table is None:
    raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
I = []
Revenue = {}
InitialInventory = {}
Demand = {}
for rec in data_table['records']:
    vals = rec['values']
    i = vals['id_number']
    if i == 'id999':
        I.append(i)
        try:
            Revenue[i] = float(vals['Revenue'])
            InitialInventory[i] = int(vals['Initial Inventory'])
            Demand[i] = int(vals['Demand'])
        except Exception as e:
            raise RuntimeError(f'Error parsing numeric fields for product {i}: {e}')
for i in I:
    if i not in Revenue or i not in InitialInventory or i not in Demand:
        raise RuntimeError(f'Missing data for product {i} in index set I.')
m = Model()
m.Params.MIPGap = 0.0001
x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
for i in I:
    m.addConstr(x_vars[i] <= InitialInventory[i], name='')
for i in I:
    m.addConstr(x_vars[i] <= Demand[i], name='')
m.setObjective(quicksum((Revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for i in I:
        print(x_vars[i].VarName, x_vars[i].X)
else:
    print(m.Status)