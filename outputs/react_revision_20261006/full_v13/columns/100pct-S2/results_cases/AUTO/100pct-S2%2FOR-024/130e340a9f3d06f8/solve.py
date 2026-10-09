CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through '
          'warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary '
          'inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided '
          'in the ‚Äúfixed_cost.csv‚Äù file. Each musician or band needs to source a certain quantity of goods from '
          'these warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse '
          "is recorded in the ‚Äútransportation_costs.csv‚Äù file. Demand information can be gained in 'demand.csv'. "
          'The objective is to determine which warehouses should be activated so that the demand of all musicians and '
          'bands is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'warehouse is operational. The decision variables x_{ij} represent the quantity of goods that musician or '
          'band S_j sources from warehouse F_i. For each musician or band, x_{ij} represents the proportion of the '
          'total supply obtained from different warehouses.',
 'relationships': [{'column_axis': {'id_column': 'customer', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Unnamed: 0', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'type': 'matrix'}],
 'route': 'FLP',
 'tables': [{'columns': ['customer', 'demand'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}}],
             'returned_rows': 3,
             'role': 'demand per musician/band',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '102.33'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '94.92'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '91.83'}}],
             'returned_rows': 3,
             'role': 'warehouse fixed costs',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'C1': '1506.22', 'C2': '70.9', 'C3': '8.44', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.44', 'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68', 'Unnamed: 0': 'S3'}}],
             'returned_rows': 3,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [3, 3],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [3, 3]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    warehouses = []
    warehouse_set = set()
    for (source_row, row) in CSVQA_FRAMES['file_1_view_0'].iterrows():
        warehouse = row['Unnamed: 0']
        if warehouse not in warehouse_set:
            warehouses.append(warehouse)
            warehouse_set.add(warehouse)
    for (source_row, row) in CSVQA_FRAMES['file_2_view_0'].iterrows():
        warehouse = row['Unnamed: 0']
        if warehouse not in warehouse_set:
            warehouses.append(warehouse)
            warehouse_set.add(warehouse)
    musicians = []
    musician_set = set()
    for (source_row, row) in CSVQA_FRAMES['file_0_view_0'].iterrows():
        customer = row['customer']
        if customer not in musician_set:
            musicians.append(customer)
            musician_set.add(customer)
    for col in ['C1', 'C2', 'C3']:
        if col not in musician_set:
            musicians.append(col)
            musician_set.add(col)
    demand = {}
    for (source_row, row) in CSVQA_FRAMES['file_0_view_0'].iterrows():
        customer = row['customer']
        try:
            demand[customer] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {customer}: {row['demand']}")
    fixed_costs = {}
    for (source_row, row) in CSVQA_FRAMES['file_1_view_0'].iterrows():
        warehouse = row['Unnamed: 0']
        try:
            fixed_costs[warehouse] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed_costs value for warehouse {warehouse}: {row['fixed_costs']}")
    transportation_costs = {}
    for (source_row, row) in CSVQA_FRAMES['file_2_view_0'].iterrows():
        warehouse = row['Unnamed: 0']
        transportation_costs[warehouse] = {}
        for musician in ['C1', 'C2', 'C3']:
            try:
                transportation_costs[warehouse][musician] = float(row[musician])
            except Exception:
                raise ValueError(f'Invalid transportation cost for warehouse {warehouse}, musician {musician}: {row[musician]}')
    M = sum((demand[j] for j in musicians if j in demand))
    for i in warehouses:
        if i not in fixed_costs:
            raise ValueError(f'Missing fixed cost for warehouse {i}')
        if i not in transportation_costs:
            raise ValueError(f'Missing transportation costs for warehouse {i}')
        for j in musicians:
            if j not in transportation_costs[i]:
                raise ValueError(f'Missing transportation cost for warehouse {i}, musician {j}')
    for j in musicians:
        if j not in demand:
            raise ValueError(f'Missing demand for musician/band {j}')
    m = gp.Model('Bandcamp_FLP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(warehouses, musicians, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((transportation_costs[i][j] * x_vars[i, j] for i in warehouses for j in musicians)) + gp.quicksum((fixed_costs[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in musicians), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in musicians)) <= M * y_vars[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)