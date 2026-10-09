CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A coffee retail store needs to allocate some types of coffee products into different retail cabinets. '
          'Specifically, the store has some cabinets, each with a capacity limit provided in "capacity.csv." The '
          'predefined value and weight of each coffee product can be found in "products.csv." The objective is to '
          'determine the optimal number of units of each coffee product to place in each cabinet to maximize the total '
          'value of the products across all cabinets, while ensuring that the total weight of the coffee in each '
          'cabinet does not exceed its capacity. The decision variables x_ij represent the number of units of coffee '
          'product j to be placed in cabinet i.The decision variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['CabinetID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'CabinetID': '1', 'Capacity': '400'}},
                         {'source_row': 1, 'values': {'CabinetID': '2', 'Capacity': '600'}},
                         {'source_row': 2, 'values': {'CabinetID': '3', 'Capacity': '500'}},
                         {'source_row': 3, 'values': {'CabinetID': '4', 'Capacity': '700'}},
                         {'source_row': 4, 'values': {'CabinetID': '5', 'Capacity': '450'}},
                         {'source_row': 5, 'values': {'CabinetID': '6', 'Capacity': '650'}},
                         {'source_row': 6, 'values': {'CabinetID': '7', 'Capacity': '550'}},
                         {'source_row': 7, 'values': {'CabinetID': '8', 'Capacity': '750'}},
                         {'source_row': 8, 'values': {'CabinetID': '9', 'Capacity': '480'}},
                         {'source_row': 9, 'values': {'CabinetID': '10', 'Capacity': '520'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 18,
             'records': [{'source_row': 0,
                          'values': {'ProductName': 'Espresso Beans', 'Value': '100', 'Weight': '1.0'}},
                         {'source_row': 1,
                          'values': {'ProductName': 'Colombian Roast', 'Value': '150', 'Weight': '1.5'}},
                         {'source_row': 2, 'values': {'ProductName': 'Arabica Blend', 'Value': '80', 'Weight': '1.2'}},
                         {'source_row': 3, 'values': {'ProductName': 'French Roast', 'Value': '120', 'Weight': '1.3'}},
                         {'source_row': 4, 'values': {'ProductName': 'Italian Roast', 'Value': '130', 'Weight': '1.4'}},
                         {'source_row': 5, 'values': {'ProductName': 'House Blend', 'Value': '110', 'Weight': '1.1'}},
                         {'source_row': 6,
                          'values': {'ProductName': 'Sumatra Coffee', 'Value': '160', 'Weight': '1.8'}},
                         {'source_row': 7, 'values': {'ProductName': 'Mocha Java', 'Value': '90', 'Weight': '1.2'}},
                         {'source_row': 8,
                          'values': {'ProductName': 'Hazelnut Flavor', 'Value': '95', 'Weight': '1.0'}},
                         {'source_row': 9, 'values': {'ProductName': 'Caramel Blend', 'Value': '105', 'Weight': '1.3'}},
                         {'source_row': 10,
                          'values': {'ProductName': 'Vanilla Flavor', 'Value': '85', 'Weight': '1.2'}},
                         {'source_row': 11,
                          'values': {'ProductName': 'Cappuccino Mix', 'Value': '140', 'Weight': '1.5'}},
                         {'source_row': 12, 'values': {'ProductName': 'Pumpkin Spice', 'Value': '75', 'Weight': '1.1'}},
                         {'source_row': 13, 'values': {'ProductName': 'Decaf Roast', 'Value': '60', 'Weight': '1.0'}},
                         {'source_row': 14,
                          'values': {'ProductName': 'Organic Roast', 'Value': '170', 'Weight': '1.6'}},
                         {'source_row': 15, 'values': {'ProductName': 'Cold Brew', 'Value': '115', 'Weight': '1.4'}},
                         {'source_row': 16,
                          'values': {'ProductName': 'Peruvian Blend', 'Value': '155', 'Weight': '1.7'}},
                         {'source_row': 17, 'values': {'ProductName': 'Kenyan AA', 'Value': '125', 'Weight': '1.3'}}],
             'returned_rows': 18,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'CabinetID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'CabinetID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'CabinetID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'CabinetID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    cabinets_df = CSVQA_FRAMES['file_0_view_0']
    products_df = CSVQA_FRAMES['file_1_view_0']
    cabinets = list(cabinets_df['CabinetID'])
    products = list(products_df['ProductName'])
    capacity = {}
    for (idx, row) in cabinets_df.iterrows():
        cab_id = row['CabinetID']
        cap_str = row['Capacity']
        try:
            cap = float(cap_str)
        except Exception:
            raise ValueError(f'Invalid Capacity for CabinetID {cab_id}: {cap_str}')
        capacity[cab_id] = cap
    value = {}
    weight = {}
    for (idx, row) in products_df.iterrows():
        prod = row['ProductName']
        val_str = row['Value']
        wgt_str = row['Weight']
        try:
            val = float(val_str)
        except Exception:
            raise ValueError(f'Invalid Value for ProductName {prod}: {val_str}')
        try:
            wgt = float(wgt_str)
        except Exception:
            raise ValueError(f'Invalid Weight for ProductName {prod}: {wgt_str}')
        value[prod] = val
        weight[prod] = wgt
    if set(cabinets) != set(capacity.keys()):
        raise ValueError('Mismatch in cabinets and capacity keys')
    if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
        raise ValueError('Mismatch in products and value/weight keys')
    m = gp.Model('CoffeeCabinetAllocation')
    quantity_vars = m.addVars(cabinets, products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * quantity_vars[i, j] for i in cabinets for j in products)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * quantity_vars[i, j] for j in products)) <= capacity[i] for i in cabinets), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')