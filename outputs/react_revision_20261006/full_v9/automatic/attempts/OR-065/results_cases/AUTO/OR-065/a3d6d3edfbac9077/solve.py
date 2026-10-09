CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'On the Bandcamp sales platform, independent musicians and bands require inventory replenishment through '
          'warehouses. Multiple distribution warehouses, located in different cities, can provide the necessary '
          'inventory. Each warehouse incurs a fixed cost when starting operations, and the fixed cost data is provided '
          'in the “fixed_cost.csv” file. Each musician or band needs to source a certain quantity of goods from these '
          'warehouses. For each musician or band, the transportation cost per unit of goods from each warehouse is '
          "recorded in the “transportation_costs.csv” file. Demand information can be gained in 'demand.csv'. The "
          'objective is to determine which warehouses should be activated so that the demand of all musicians and '
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
             'records': [{'source_row': 0,
                          'values': {'C1': '1506.22', 'C2': '70.90000000000001', 'C3': '8.44', 'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '1732.65', 'C2': '1780.72', 'C3': '567.4400000000001', 'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '115.66', 'C2': '100.76', 'C3': '64.68000000000001', 'Unnamed: 0': 'S3'}}],
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
    demand_df = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_df = CSVQA_FRAMES['file_1_view_0']
    trans_cost_df = CSVQA_FRAMES['file_2_view_0']
    warehouses = list(fixed_cost_df['Unnamed: 0'])
    customers = list(demand_df['customer'])
    for col in trans_cost_df.columns:
        if col != 'Unnamed: 0' and col not in customers:
            customers.append(col)
    demand = {}
    for (_, row) in demand_df.iterrows():
        cust = row['customer']
        try:
            demand[cust] = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cust}: {row['demand']}")
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        wh = row['Unnamed: 0']
        try:
            fixed_cost[wh] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed cost for warehouse {wh}: {row['fixed_costs']}")
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        wh = row['Unnamed: 0']
        cost[wh] = {}
        for cust in customers:
            if cust == 'Unnamed: 0':
                continue
            if cust in trans_cost_df.columns:
                val = row[cust]
                try:
                    cost[wh][cust] = float(val)
                except Exception:
                    raise ValueError(f'Non-numeric transportation cost for warehouse {wh}, customer {cust}: {val}')
            else:
                raise ValueError(f'Missing transportation cost for warehouse {wh}, customer {cust}')
    M = sum((demand[j] for j in customers))
    for wh in warehouses:
        if wh not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {wh}')
        if wh not in cost:
            raise ValueError(f'Missing transportation cost row for warehouse {wh}')
        for cust in customers:
            if cust not in cost[wh]:
                raise ValueError(f'Missing transportation cost for warehouse {wh}, customer {cust}')
    for cust in customers:
        if cust not in demand:
            raise ValueError(f'Missing demand for customer {cust}')
    m = gp.Model('Bandcamp_FLP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(warehouses, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in customers)) + gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in customers)) <= M * y_vars[i] for i in warehouses), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)