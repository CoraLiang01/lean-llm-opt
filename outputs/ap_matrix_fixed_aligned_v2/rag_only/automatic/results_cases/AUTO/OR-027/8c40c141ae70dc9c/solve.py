CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of products with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘Organ’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i indicate '
          'the number of units of each ‘Organ’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Sub Category', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SupermartGrocerySales-RetailAnalyticsDataset.csv',
             'filters': {'conditions': [{'column': 'Sub Category',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Organ’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Organic'}],
                         'logic': 'or'},
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
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    data = [{'Sub Category': 'Organic Fruits', 'Revenue': 60.8, 'Initial Inventory': 5034020.0, 'Demand': 678906}, {'Sub Category': 'Organic Staples', 'Revenue': 918.45, 'Initial Inventory': 5589290.0, 'Demand': 749927}, {'Sub Category': 'Organic Vegetables', 'Revenue': 77.52, 'Initial Inventory': 5202710.0, 'Demand': 699808}]
    I = [rec['Sub Category'] for rec in data]
    Revenue = {rec['Sub Category']: float(rec['Revenue']) for rec in data}
    Inventory = {rec['Sub Category']: float(rec['Initial Inventory']) for rec in data}
    Demand = {rec['Sub Category']: float(rec['Demand']) for rec in data}
    for i in I:
        if i not in Revenue or i not in Inventory or i not in Demand:
            raise ValueError(f'Missing data for product {i}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(quicksum((Revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x_vars[i] <= Inventory[i], name='')
        m.addConstr(x_vars[i] <= Demand[i], name='')
        m.addConstr(x_vars[i] >= 0, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for i in I:
            print(x_vars[i].VarName, x_vars[i].X)
    else:
        print(m.Status)
    return m
m = build_and_solve()