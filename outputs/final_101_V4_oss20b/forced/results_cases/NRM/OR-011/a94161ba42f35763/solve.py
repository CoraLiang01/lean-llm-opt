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
 'tables': [{'aliases': ['file_0_view_0', 'OnlineRetailSalesDataset.csv'],
             'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': 'id999',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
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
import gurobipy as gp
from gurobipy import GRB
table = CSVQA_DATA['tables'][0]
records = table['records']
ids = []
w = {}
d = {}
c = {}
for rec in records:
    vals = rec['values']
    id_num = vals['id_number']
    ids.append(id_num)
    w[id_num] = float(vals['Revenue'])
    d[id_num] = int(vals['Demand'])
    c[id_num] = int(vals['Initial Inventory'])
m = gp.Model()
m.setParam('MIPGap', 1e-06)
x = m.addVars(ids, vtype=GRB.INTEGER, lb=0)
for i in ids:
    m.addConstr(x[i] <= c[i])
    m.addConstr(x[i] <= d[i])
m.setObjective(gp.quicksum((w[i] * x[i] for i in ids)), GRB.MAXIMIZE)
m.optimize()
if m.status == GRB.OPTIMAL:
    print(m.objVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.status)