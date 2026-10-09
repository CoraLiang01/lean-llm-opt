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
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '1083'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '776'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '16214'}}],
             'returned_rows': 3,
             'role': 'demand per customer',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'filters': {'conditions': [], 'logic': 'and'},
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

def solve_problem():
    demand_df = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_df = CSVQA_FRAMES['file_1_view_0']
    cost_df = CSVQA_FRAMES['file_2_view_0']
    I = [row['Unnamed: 0'] for row in fixed_cost_df.to_dict('records')]
    J = [row['customer'] for row in demand_df.to_dict('records')]
    demand = {}
    for row in demand_df.to_dict('records'):
        cust = row['customer']
        try:
            demand[cust] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")
    fixed_costs = {}
    for row in fixed_cost_df.to_dict('records'):
        wh = row['Unnamed: 0']
        try:
            fixed_costs[wh] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed cost for warehouse {wh}: {row['fixed_costs']}")
    cost = {}
    for row in cost_df.to_dict('records'):
        i = row['Unnamed: 0']
        cost[i] = {}
        for j in J:
            try:
                cost[i][j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid transportation cost for warehouse {i}, customer {j}: {row[j]}')
    M = sum((demand[j] for j in J))
    if set(I) != set(cost.keys()):
        raise ValueError('Mismatch between warehouse indices in fixed_costs and transportation_costs.')
    for i in I:
        if set(J) != set(cost[i].keys()):
            raise ValueError(f'Mismatch between customer indices in transportation_costs for warehouse {i}.')
    if set(J) != set(demand.keys()):
        raise ValueError('Mismatch between customer indices in demand and transportation_costs.')
    m = gp.Model('Bandcamp_FLP')
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * quantity_vars[i, j] for i in I for j in J)) + gp.quicksum((fixed_costs[i] * activation_vars[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) == demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for j in J)) <= M * activation_vars[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()