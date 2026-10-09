CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'multiple supermarkets require inventory, and several suppliers located in different cities can provide the '
          'necessary goods. Each supplier incurs a fixed cost upon starting operations, with the fixed cost data '
          'provided in "fixed_cost.csv." Each supermarket needs to source a unit good from these suppliers. For each '
          'supermarket, the transportation cost per unit of goods from each supplier is recorded in '
          '"transportation_costs.csv." Demand information can be gained in \'demand.csv\'. The objective is to '
          'determine which suppliers to open so that the demand of all supermarkets is met while minimizing the total '
          'cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The '
          'decision variables x_ij represent the quantity of goods that supermarket S_j sources from supplier F_i. For '
          'each supermarket, x_ij represents the proportion of the total supply obtained from different suppliers. '
          'These decision variables help determine the optimal allocation of supply to minimize the total of fixed and '
          'transportation costs.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1097'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '61'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '11'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '7'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '82'}},
                         {'source_row': 5, 'values': {'customer': 'C6', 'demand': '37'}},
                         {'source_row': 6, 'values': {'customer': 'C7', 'demand': '483'}},
                         {'source_row': 7, 'values': {'customer': 'C8', 'demand': '582'}},
                         {'source_row': 8, 'values': {'customer': 'C9', 'demand': '223'}},
                         {'source_row': 9, 'values': {'customer': 'C10', 'demand': '89'}},
                         {'source_row': 10, 'values': {'customer': 'C11', 'demand': '60'}},
                         {'source_row': 11, 'values': {'customer': 'C12', 'demand': '55'}}],
             'returned_rows': 12,
             'role': 'supermarket demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}},
                         {'source_row': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}},
                         {'source_row': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '97.05'}},
                         {'source_row': 7, 'values': {'Unnamed: 0': 'S8', 'fixed_costs': '103'}},
                         {'source_row': 8, 'values': {'Unnamed: 0': 'S9', 'fixed_costs': '90.45'}},
                         {'source_row': 9, 'values': {'Unnamed: 0': 'S10', 'fixed_costs': '96.73'}},
                         {'source_row': 10, 'values': {'Unnamed: 0': 'S11', 'fixed_costs': '96.43000000000001'}},
                         {'source_row': 11, 'values': {'Unnamed: 0': 'S12', 'fixed_costs': '112.19'}}],
             'returned_rows': 12,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'C1': '284.11',
                                     'C10': '8.869999999999999',
                                     'C11': '1129.47',
                                     'C12': '185.53',
                                     'C2': '53.78',
                                     'C3': '10.62',
                                     'C4': '111.27',
                                     'C5': '158.5',
                                     'C6': '8.789999999999999',
                                     'C7': '53.79',
                                     'C8': '8.84',
                                     'C9': '1911.43',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '7.19',
                                     'C10': '1.45',
                                     'C11': '49.14',
                                     'C12': '0.05',
                                     'C2': '1031.96',
                                     'C3': '90.94',
                                     'C4': '276.97',
                                     'C5': '0.45',
                                     'C6': '0.2',
                                     'C7': '49.14',
                                     'C8': '1.05',
                                     'C9': '2079.54',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '151.1',
                                     'C10': '1.63',
                                     'C11': '884.47',
                                     'C12': '0.96',
                                     'C2': '884.48',
                                     'C3': '4.33',
                                     'C4': '277.04',
                                     'C5': '0.33',
                                     'C6': '0.19',
                                     'C7': '49.14',
                                     'C8': '0.99',
                                     'C9': '99.03',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '144.16',
                                     'C10': '19.74',
                                     'C11': '868.74',
                                     'C12': '19.85',
                                     'C2': '868.75',
                                     'C3': '94.2',
                                     'C4': '285.48',
                                     'C5': '16.93',
                                     'C6': '0.9399999999999999',
                                     'C7': '868.78',
                                     'C8': '16.6',
                                     'C9': '98.69',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '151.34',
                                     'C10': '0.84',
                                     'C11': '883.6',
                                     'C12': '0.58',
                                     'C2': '1030.88',
                                     'C3': '91.43000000000001',
                                     'C4': '13.24',
                                     'C5': '0.72',
                                     'C6': '0.87',
                                     'C7': '49.09',
                                     'C8': '0.01',
                                     'C9': '99.05',
                                     'Unnamed: 0': 'S5'}},
                         {'source_row': 5,
                          'values': {'C1': '7.18',
                                     'C10': '1.06',
                                     'C11': '884.3099999999999',
                                     'C12': '0.34',
                                     'C2': '49.13',
                                     'C3': '90.72',
                                     'C4': '277.57',
                                     'C5': '0.37',
                                     'C6': '0.58',
                                     'C7': '1031.74',
                                     'C8': '0.76',
                                     'C9': '1782.98',
                                     'Unnamed: 0': 'S6'}},
                         {'source_row': 6,
                          'values': {'C1': '104.38',
                                     'C10': '2080.99',
                                     'C11': '1545.08',
                                     'C12': '99.06999999999999',
                                     'C2': '1324.35',
                                     'C3': '1829.39',
                                     'C4': '1857.57',
                                     'C5': '1782.69',
                                     'C6': '2079.47',
                                     'C7': '1324.31',
                                     'C8': '2080.29',
                                     'C9': '0',
                                     'Unnamed: 0': 'S7'}},
                         {'source_row': 7,
                          'values': {'C1': '129.51',
                                     'C10': '1.69',
                                     'C11': '49.14',
                                     'C12': '0.05',
                                     'C2': '1031.96',
                                     'C3': '4.33',
                                     'C4': '276.97',
                                     'C5': '0.02',
                                     'C6': '0.23',
                                     'C7': '884.5599999999999',
                                     'C8': '1.22',
                                     'C9': '2079.54',
                                     'Unnamed: 0': 'S8'}},
                         {'source_row': 8,
                          'values': {'C1': '50.93',
                                     'C10': '47.59',
                                     'C11': '5.75',
                                     'C12': '999.9400000000001',
                                     'C2': '5.75',
                                     'C3': '1057.85',
                                     'C4': '58.62',
                                     'C5': '47.63',
                                     'C6': '1000.41',
                                     'C7': '103.48',
                                     'C8': '47.6',
                                     'C9': '1642.85',
                                     'Unnamed: 0': 'S9'}},
                         {'source_row': 9,
                          'values': {'C1': '129.62',
                                     'C10': '0.08',
                                     'C11': '49.13',
                                     'C12': '0.04',
                                     'C2': '884.35',
                                     'C3': '91.09999999999999',
                                     'C4': '277.12',
                                     'C5': '0.27',
                                     'C6': '0.07000000000000001',
                                     'C7': '1031.78',
                                     'C8': '0.91',
                                     'C9': '99.03',
                                     'Unnamed: 0': 'S10'}},
                         {'source_row': 10,
                          'values': {'C1': '53.3',
                                     'C10': '49.1',
                                     'C11': '0.08',
                                     'C12': '49.12',
                                     'C2': '0',
                                     'C3': '941.91',
                                     'C4': '58.92',
                                     'C5': '1031.61',
                                     'C6': '49.13',
                                     'C7': '0.03',
                                     'C8': '1030.99',
                                     'C9': '1324.29',
                                     'Unnamed: 0': 'S11'}},
                         {'source_row': 11,
                          'values': {'C1': '959.55',
                                     'C10': '49.1',
                                     'C11': '0.12',
                                     'C12': '1031.53',
                                     'C2': '0.11',
                                     'C3': '941.98',
                                     'C4': '1237.42',
                                     'C5': '49.13',
                                     'C6': '1031.86',
                                     'C7': '0.09',
                                     'C8': '1031.07',
                                     'C9': '73.56999999999999',
                                     'Unnamed: 0': 'S12'}}],
             'returned_rows': 12,
             'role': 'supplier-supermarket transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [12, 12],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [12, 12]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    suppliers = []
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        suppliers.append(supplier)
    for (_, row) in cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            suppliers.append(supplier)
    supermarkets = []
    for (_, row) in demand_frame.iterrows():
        customer = row['customer']
        supermarkets.append(customer)
    for col in cost_frame.columns:
        if col != 'Unnamed: 0' and col not in supermarkets:
            supermarkets.append(col)
    suppliers = list(dict.fromkeys(suppliers))
    supermarkets = list(dict.fromkeys(supermarkets))
    demand = {}
    for (_, row) in demand_frame.iterrows():
        customer = row['customer']
        demand[customer] = float(row['demand'])
    fixed_cost = {}
    for (_, row) in fixed_cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        fixed_cost[supplier] = float(row['fixed_costs'])
    cost = {}
    for (_, row) in cost_frame.iterrows():
        supplier = row['Unnamed: 0']
        cost[supplier] = {}
        for customer in supermarkets:
            if customer == 'Unnamed: 0':
                continue
            val = row.get(customer, '')
            if val == '' or val is None:
                raise ValueError(f'Missing transportation cost for supplier {supplier}, customer {customer}')
            cost[supplier][customer] = float(val)
    U = {}
    for supplier in suppliers:
        U[supplier] = {}
        for customer in supermarkets:
            if customer not in demand:
                raise ValueError(f'Missing demand for customer {customer}')
            U[supplier][customer] = demand[customer]
    for supplier in suppliers:
        if supplier not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {supplier}')
        if supplier not in cost:
            raise ValueError(f'Missing cost row for supplier {supplier}')
        for customer in supermarkets:
            if customer not in cost[supplier]:
                raise ValueError(f'Missing cost for supplier {supplier}, customer {customer}')
            if customer not in demand:
                raise ValueError(f'Missing demand for customer {customer}')
    m = gp.Model('FLP')
    m.setParam('MIPGap', 0.0001)
    x_keys = [(i, j) for i in suppliers for j in supermarkets]
    x_vars = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in suppliers for j in supermarkets)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in suppliers)), GRB.MINIMIZE)
    for j in supermarkets:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
    for i in suppliers:
        for j in supermarkets:
            m.addConstr(x_vars[i, j] <= U[i][j] * y_vars[i], name=f'activation_{i}_{j}')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')