CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket needs to replenish its stock, and the ‚Äòproducts.csv‚Äô provides the relevant income '
          'statement for each type of produce (e.g. leafy greens, mushrooms, etc.). The supermarket also has an '
          'overall stock capacity constraint detailed in ‚Äòcapacity.csv‚Äô.The objective is to decide which products '
          'to order each day and the quantities to be ordered in order to maximise the overall benefits while adhering '
          'to the overall stock capacity. The decision variable x_i represents the number of units of the ith product '
          'to be ordered each day.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '875'}}],
             'returned_rows': 1,
             'role': 'overall stock capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Weight', 'Value'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Spinach', 'Value': '64', 'Weight': '230'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Shiitake Mushrooms', 'Value': '75', 'Weight': '637'}},
                         {'source_row': 2, 'values': {'ProductName': 'Apples', 'Value': '68', 'Weight': '773'}},
                         {'source_row': 3, 'values': {'ProductName': 'Carrots', 'Value': '11', 'Weight': '653'}},
                         {'source_row': 4, 'values': {'ProductName': 'Basil', 'Value': '91', 'Weight': '755'}},
                         {'source_row': 5, 'values': {'ProductName': 'Potatoes', 'Value': '31', 'Weight': '670'}},
                         {'source_row': 6, 'values': {'ProductName': 'Green Beans', 'Value': '90', 'Weight': '505'}},
                         {'source_row': 7, 'values': {'ProductName': 'Blueberries', 'Value': '56', 'Weight': '821'}},
                         {'source_row': 8, 'values': {'ProductName': 'Oranges', 'Value': '10', 'Weight': '83'}},
                         {'source_row': 9, 'values': {'ProductName': 'Watermelons', 'Value': '24', 'Weight': '249'}}],
             'returned_rows': 10,
             'role': 'product income statement and attributes',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_df = CSVQA_FRAMES['file_1_view_0']
    capacity_df = CSVQA_FRAMES['file_0_view_0']
    product_names = products_df['ProductName'].tolist()
    try:
        values = products_df['Value'].astype(float).tolist()
        weights = products_df['Weight'].astype(float).tolist()
    except Exception as e:
        raise ValueError(f'Error converting product Value/Weight to float: {e}')
    value_dict = dict(zip(product_names, values))
    weight_dict = dict(zip(product_names, weights))
    try:
        C = float(capacity_df['Capacity'].iloc[0])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    m = gp.Model('Supermarket_Stock_Replenishment')
    quantity_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[i] * quantity_vars[i] for i in product_names)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * quantity_vars[i] for i in product_names)) <= C, name='stock_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()