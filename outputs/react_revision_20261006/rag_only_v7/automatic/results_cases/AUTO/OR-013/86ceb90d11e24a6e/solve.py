CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'This supermarket offers a variety of best-selling products, and the specific revenue information is shown '
          'in the table, with relevant data provided in the “Revenue” column. Each product has its own demand level. '
          'The retailer’s objective is to maximize the total revenue by focusing on the sales volume of “4U” products. '
          'The initial inventory levels of these products are detailed in the “Initial Inventory” column. During the '
          'sales period, restocking is not allowed and there is no in-transit inventory.\n'
          '\n'
          'Demand for the “4U” products during the sales horizon is assumed to be deterministic and known in advance, '
          'with demand quantities specified in the “Demand” column. The decision variable x_i represents the number of '
          'units of each “4U” product i that the retailer plans to fulfill, where each x_i is a non-negative integer. '
          'Because the fulfillment quantity cannot exceed either the available inventory or the realized demand, the '
          'decision variables must satisfy both inventory and demand constraints.\n'
          '\n'
          'The retailer therefore aims to determine the optimal fulfillment quantities in order to maximize total '
          'revenue while respecting both inventory availability and demand limits.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineSalesinUSA.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'sales volume of “4U” products',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': '4U'}],
                         'logic': 'and'},
             'original_rows': 47932,
             'records': [{'source_row': 1,
                          'values': {'Demand': '5',
                                     'Initial Inventory': '30',
                                     'Product Name': '4U_Service_22',
                                     'Revenue': '56.0'}},
                         {'source_row': 2,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_36',
                                     'Revenue': '21.6'}},
                         {'source_row': 3,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_7',
                                     'Revenue': '62.5'}}],
             'returned_rows': 3,
             'role': 'revenue management product data',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem():
    df = CSVQA_FRAMES['file_0_view_0']
    mask = df['Product Name'].str.casefold().str.startswith('4u')
    df_4u = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_4u.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(df_4u['Product Name'])
    try:
        r = df_4u.set_index('Product Name')['Revenue'].astype(float).to_dict()
        d = df_4u.set_index('Product Name')['Demand'].astype(float).to_dict()
        s = df_4u.set_index('Product Name')['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting parameter columns to float: {e}')
    for i in I:
        if i not in r or i not in d or i not in s:
            raise ValueError(f'Missing parameter for product {i}')
    m = gp.Model('4U_Fulfillment')
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, obj=0, name='')
    for i in I:
        m.addConstr(quantity_vars[i] <= d[i], name='')
        m.addConstr(quantity_vars[i] <= s[i], name='')
    m.setObjective(gp.quicksum((r[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for i in I:
            v = quantity_vars[i]
            print(f'{v.VarName} {v.X}')
    else:
        print(m.Status)
    return m
m = solve_problem()