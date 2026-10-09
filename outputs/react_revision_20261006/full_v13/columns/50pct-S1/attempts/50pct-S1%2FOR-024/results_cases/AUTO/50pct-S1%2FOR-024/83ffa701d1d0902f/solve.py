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
             'role': 'musician/band demand',
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
             'role': 'warehouse fixed cost',
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

def solve_problem(CSVQA_FRAMES):
    demand_frame = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_frame = CSVQA_FRAMES['file_1_view_0']
    cost_frame = CSVQA_FRAMES['file_2_view_0']
    warehouses = []
    warehouses_set = set()
    for (_, row) in fixed_cost_frame.iterrows():
        wid = row['Unnamed: 0']
        if wid not in warehouses_set:
            warehouses.append(wid)
            warehouses_set.add(wid)
    for (_, row) in cost_frame.iterrows():
        wid = row['Unnamed: 0']
        if wid not in warehouses_set:
            warehouses.append(wid)
            warehouses_set.add(wid)
    musicians = []
    musicians_set = set()
    for (_, row) in demand_frame.iterrows():
        cid = row['customer']
        if cid not in musicians_set:
            musicians.append(cid)
            musicians_set.add(cid)
    for col in cost_frame.columns:
        if col != 'Unnamed: 0' and col not in musicians_set:
            musicians.append(col)
            musicians_set.add(col)
    demand = {}
    for (_, row) in demand_frame.iterrows():
        cid = row['customer']
        try:
            demand[cid] = float(row['demand'])
        except Exception:
            raise ValueError(f"Non-numeric demand for customer {cid}: {row['demand']}")
    fixed_cost = {}
    for (_, row) in fixed_cost_frame.iterrows():
        wid = row['Unnamed: 0']
        try:
            fixed_cost[wid] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Non-numeric fixed_costs for warehouse {wid}: {row['fixed_costs']}")
    cost = {}
    for (_, row) in cost_frame.iterrows():
        wid = row['Unnamed: 0']
        cost[wid] = {}
        for cid in musicians:
            if cid in row:
                try:
                    cost[wid][cid] = float(row[cid])
                except Exception:
                    raise ValueError(f'Non-numeric transportation cost for warehouse {wid}, customer {cid}: {row[cid]}')
            else:
                raise ValueError(f'Missing transportation cost for warehouse {wid}, customer {cid}')
    M = sum((demand[cid] for cid in musicians))
    for wid in warehouses:
        if wid not in fixed_cost:
            raise ValueError(f'Missing fixed cost for warehouse {wid}')
        if wid not in cost:
            raise ValueError(f'Missing cost row for warehouse {wid}')
        for cid in musicians:
            if cid not in cost[wid]:
                raise ValueError(f'Missing cost for warehouse {wid}, customer {cid}')
    for cid in musicians:
        if cid not in demand:
            raise ValueError(f'Missing demand for customer {cid}')
    m = gp.Model('Bandcamp_FLP')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(warehouses, musicians, lb=0, vtype=GRB.CONTINUOUS, name='')
    y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[wid][cid] * x_vars[wid, cid] for wid in warehouses for cid in musicians)) + gp.quicksum((fixed_cost[wid] * y_vars[wid] for wid in warehouses)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[wid, cid] for wid in warehouses)) == demand[cid] for cid in musicians), name='')
    m.addConstrs((gp.quicksum((x_vars[wid, cid] for cid in musicians)) <= M * y_vars[wid] for wid in warehouses), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')