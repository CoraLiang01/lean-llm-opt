CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'aliases': ['file_0_view_0', 'SupermartGrocerySales-RetailAnalyticsDataset.csv'],
             'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [{'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': 'classified under ‘Organ’',
                                         'inclusive': 'both',
                                         'operator': 'contains',
                                         'value': 'Organ'}],
                         'logic': 'and'},
             'original_rows': 23,
             'records': [{'source_row': 17,
                          'values': {'Demand': '678906',
                                     'Initial Inventory': '5034020.0',
                                     'Revenue': '60.8',
                                     'Sub Category': 'Organic Fruits'}},
                         {'source_row': 18,
                          'values': {'Demand': '749927',
                                     'Initial Inventory': '5589290.0',
                                     'Revenue': '918.45',
                                     'Sub Category': 'Organic Staples'}},
                         {'source_row': 19,
                          'values': {'Demand': '699808',
                                     'Initial Inventory': '5202710.0',
                                     'Revenue': '77.52',
                                     'Sub Category': 'Organic Vegetables'}}],
             'returned_rows': 3,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = CSVQA_DATA['tables'][0]
records = table['records']
items = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
    vals = rec['values']
    item = vals['Sub Category']
    items.append(item)
    revenue[item] = float(vals['Revenue'])
    demand[item] = int(vals['Demand'])
    inventory[item] = float(vals['Initial Inventory'])
m = gp.Model('max_revenue')
x = {}
for item in items:
    x[item] = m.addVar(vtype=GRB.INTEGER, lb=0)
for item in items:
    m.addConstr(x[item] <= demand[item])
    m.addConstr(x[item] <= inventory[item])
m.setObjective(gp.quicksum((revenue[item] * x[item] for item in items)), GRB.MAXIMIZE)
m.Params.MIPGap = 1e-06
m.optimize()
if m.status == GRB.OPTIMAL:
    print(m.ObjVal)
    for var in m.getVars():
        print(var.VarName, var.X)
else:
    print(m.status)