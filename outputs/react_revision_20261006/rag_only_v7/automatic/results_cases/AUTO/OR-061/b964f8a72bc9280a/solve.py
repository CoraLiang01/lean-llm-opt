CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the “fixed_cost.csv” file. Each branch needs to source a '
          'certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of goods '
          'from each supplier is recorded in the “transportation_costs.csv” file. Demand information can be gained in '
          "'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all branches "
          'is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a '
          'supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
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
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer': 'C1', 'demand': '143'}},
                         {'source_row': 1, 'values': {'customer': 'C2', 'demand': '6'}},
                         {'source_row': 2, 'values': {'customer': 'C3', 'demand': '10'}},
                         {'source_row': 3, 'values': {'customer': 'C4', 'demand': '25'}},
                         {'source_row': 4, 'values': {'customer': 'C5', 'demand': '3'}}],
             'returned_rows': 5,
             'role': 'branch demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Unnamed: 0', 'fixed_costs'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '97.65000000000001'}},
                         {'source_row': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.76000000000001'}},
                         {'source_row': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '100.76'}},
                         {'source_row': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '105.32'}},
                         {'source_row': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '98.88'}}],
             'returned_rows': 5,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'C1': '150.74',
                                     'C2': '0.02',
                                     'C3': '49.13',
                                     'C4': '2080.15',
                                     'C5': '426.4',
                                     'Unnamed: 0': 'S1'}},
                         {'source_row': 1,
                          'values': {'C1': '233.05',
                                     'C2': '97.73',
                                     'C3': '49.84',
                                     'C4': '1982.39',
                                     'C5': '23.96',
                                     'Unnamed: 0': 'S2'}},
                         {'source_row': 2,
                          'values': {'C1': '55.68',
                                     'C2': '935.61',
                                     'C3': '4.03',
                                     'C4': '73.09',
                                     'C5': '525.3200000000001',
                                     'Unnamed: 0': 'S3'}},
                         {'source_row': 3,
                          'values': {'C1': '1483.82',
                                     'C2': '1801.08',
                                     'C3': '112.16',
                                     'C4': '816.05',
                                     'C5': '107.01',
                                     'Unnamed: 0': 'S4'}},
                         {'source_row': 4,
                          'values': {'C1': '1119.47',
                                     'C2': '884.3099999999999',
                                     'C3': '0.08',
                                     'C4': '1544.95',
                                     'C5': '543.67',
                                     'Unnamed: 0': 'S5'}}],
             'returned_rows': 5,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [5, 5],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [5, 5]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def solve_problem():
    demand_df = CSVQA_FRAMES['file_0_view_0']
    fixed_cost_df = CSVQA_FRAMES['file_1_view_0']
    trans_cost_df = CSVQA_FRAMES['file_2_view_0']
    F1 = list(fixed_cost_df['Unnamed: 0'])
    F2 = list(trans_cost_df['Unnamed: 0'])
    F = list(dict.fromkeys(F1 + F2))
    C1 = list(demand_df['customer'])
    C2 = [col for col in trans_cost_df.columns if col in ['C1', 'C2', 'C3', 'C4', 'C5']]
    C = list(dict.fromkeys(C1 + C2))
    fixed_costs = {}
    for (idx, row) in fixed_cost_df.iterrows():
        i = row['Unnamed: 0']
        try:
            fixed_costs[i] = float(row['fixed_costs'])
        except Exception:
            raise ValueError(f"Invalid fixed_costs for supplier {i}: {row['fixed_costs']}")
    demands = {}
    for (idx, row) in demand_df.iterrows():
        j = row['customer']
        try:
            demands[j] = float(row['demand'])
        except Exception:
            raise ValueError(f"Invalid demand for customer {j}: {row['demand']}")
    trans_costs = {}
    for (idx, row) in trans_cost_df.iterrows():
        i = row['Unnamed: 0']
        for j in C2:
            try:
                trans_costs[i, j] = float(row[j])
            except Exception:
                raise ValueError(f'Invalid transportation cost for ({i},{j}): {row[j]}')
    missing_f = [i for i in F if i not in fixed_costs]
    if missing_f:
        raise ValueError(f'Missing fixed costs for suppliers: {missing_f}')
    missing_c = [j for j in C if j not in demands]
    if missing_c:
        raise ValueError(f'Missing demand for customers: {missing_c}')
    missing_tc = [(i, j) for i in F for j in C if (i, j) not in trans_costs]
    if missing_tc:
        raise ValueError(f'Missing transportation costs for supplier-customer pairs: {missing_tc}')
    m = gp.Model('facility_location')
    y_vars = m.addVars(F, vtype=GRB.BINARY, lb=0, ub=1, name='')
    x_keys = [(i, j) for i in F for j in C]
    x_vars = m.addVars(x_keys, vtype=GRB.CONTINUOUS, lb=0, name='')
    m.setObjective(gp.quicksum((fixed_costs[i] * y_vars[i] for i in F)) + gp.quicksum((trans_costs[i, j] * x_vars[i, j] for i in F for j in C)), GRB.MINIMIZE)
    for j in C:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in F)) == demands[j], name=f'demand_{j}')
    for i in F:
        for j in C:
            m.addConstr(x_vars[i, j] <= demands[j] * y_vars[i], name=f'supply_link_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(m.ObjVal)
    for v in m.getVars():
        print(v.VarName, v.X)
else:
    print(m.Status)