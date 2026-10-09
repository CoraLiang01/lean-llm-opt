CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In the Superstore chain, multiple branches require inventory replenishment, and several suppliers located '
          'in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting '
          'operations, with the fixed cost data provided in the ‚Äúfixed_cost.csv‚Äù file. Each branch needs to source '
          'a certain quantity of goods from these suppliers. For each branch, the transportation cost per unit of '
          'goods from each supplier is recorded in the ‚Äútransportation_costs.csv‚Äù file. Demand information can be '
          "gained in 'demand.csv'. The objective is to determine which suppliers to activate so that the demand of all "
          'branches is met while minimizing the total cost. The decision variables y_i are binary, indicating whether '
          'a supplier is operational (open). The decision variables x_{ij} represent the quantity of goods that branch '
          'S_j sources from supplier F_i. For each branch, x_{ij} represents the proportion of the total supply '
          'obtained from different suppliers. These decision variables help determine the optimal allocation of supply '
          'to minimize the total of fixed and transportation costs.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['customer_id', 'demand_units'],
             'file_index': 0,
             'file_name': 'demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'customer_id': 'C1', 'demand_units': '143'}},
                         {'source_row': 1, 'values': {'customer_id': 'C2', 'demand_units': '6'}},
                         {'source_row': 2, 'values': {'customer_id': 'C3', 'demand_units': '10'}},
                         {'source_row': 3, 'values': {'customer_id': 'C4', 'demand_units': '25'}},
                         {'source_row': 4, 'values': {'customer_id': 'C5', 'demand_units': '3'}}],
             'returned_rows': 5,
             'role': 'branch demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['facility_id', 'fixed_opening_cost'],
             'file_index': 1,
             'file_name': 'fixed_cost.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0, 'values': {'facility_id': 'S1', 'fixed_opening_cost': '97.65'}},
                         {'source_row': 1, 'values': {'facility_id': 'S2', 'fixed_opening_cost': '99.76'}},
                         {'source_row': 2, 'values': {'facility_id': 'S3', 'fixed_opening_cost': '100.76'}},
                         {'source_row': 3, 'values': {'facility_id': 'S4', 'fixed_opening_cost': '105.32'}},
                         {'source_row': 4, 'values': {'facility_id': 'S5', 'fixed_opening_cost': '98.88'}}],
             'returned_rows': 5,
             'role': 'supplier fixed cost',
             'table_id': 'file_1_view_0'},
            {'columns': ['facility_id',
                         'transportation_cost_to_C1',
                         'transportation_cost_to_C2',
                         'transportation_cost_to_C3',
                         'transportation_cost_to_C4',
                         'transportation_cost_to_C5'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 5,
             'records': [{'source_row': 0,
                          'values': {'facility_id': 'S1',
                                     'transportation_cost_to_C1': '150.74',
                                     'transportation_cost_to_C2': '0.02',
                                     'transportation_cost_to_C3': '49.13',
                                     'transportation_cost_to_C4': '2080.15',
                                     'transportation_cost_to_C5': '426.4'}},
                         {'source_row': 1,
                          'values': {'facility_id': 'S2',
                                     'transportation_cost_to_C1': '233.05',
                                     'transportation_cost_to_C2': '97.73',
                                     'transportation_cost_to_C3': '49.84',
                                     'transportation_cost_to_C4': '1982.39',
                                     'transportation_cost_to_C5': '23.96'}},
                         {'source_row': 2,
                          'values': {'facility_id': 'S3',
                                     'transportation_cost_to_C1': '55.68',
                                     'transportation_cost_to_C2': '935.61',
                                     'transportation_cost_to_C3': '4.03',
                                     'transportation_cost_to_C4': '73.09',
                                     'transportation_cost_to_C5': '525.32'}},
                         {'source_row': 3,
                          'values': {'facility_id': 'S4',
                                     'transportation_cost_to_C1': '1483.82',
                                     'transportation_cost_to_C2': '1801.08',
                                     'transportation_cost_to_C3': '112.16',
                                     'transportation_cost_to_C4': '816.05',
                                     'transportation_cost_to_C5': '107.01'}},
                         {'source_row': 4,
                          'values': {'facility_id': 'S5',
                                     'transportation_cost_to_C1': '1119.47',
                                     'transportation_cost_to_C2': '884.31',
                                     'transportation_cost_to_C3': '0.08',
                                     'transportation_cost_to_C4': '1544.95',
                                     'transportation_cost_to_C5': '543.67'}}],
             'returned_rows': 5,
             'role': 'supplier-branch transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    demand_df = CSVQA_FRAMES['file_0_view_0']
    demand_df = demand_df.copy()
    demand_df['demand_units'] = demand_df['demand_units'].astype(float)
    J = list(demand_df['customer_id'])
    d_j = dict(zip(demand_df['customer_id'], demand_df['demand_units']))
    fixed_cost_df = CSVQA_FRAMES['file_1_view_0']
    fixed_cost_df = fixed_cost_df.copy()
    fixed_cost_df['fixed_opening_cost'] = fixed_cost_df['fixed_opening_cost'].astype(float)
    I = list(fixed_cost_df['facility_id'])
    f_i = dict(zip(fixed_cost_df['facility_id'], fixed_cost_df['fixed_opening_cost']))
    trans_cost_df = CSVQA_FRAMES['file_2_view_0']
    trans_cost_df = trans_cost_df.copy()
    c_ij = {}
    for (idx, row) in trans_cost_df.iterrows():
        i = row['facility_id']
        c_ij[i] = {}
        for j in J:
            col = f'transportation_cost_to_{j}'
            c_ij[i][j] = float(row[col])
    if set(I) != set(f_i.keys()):
        raise ValueError('Mismatch in supplier index set and fixed cost keys')
    for i in I:
        if i not in c_ij:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        if set(J) != set(c_ij[i].keys()):
            raise ValueError(f'Missing transportation cost columns for supplier {i}')
    if set(J) != set(d_j.keys()):
        raise ValueError('Mismatch in branch index set and demand keys')
    D_j = d_j.copy()
    m = gp.Model('Superstore_FLP')
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
    activation_vars = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i][j] * quantity_vars[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * activation_vars[i] for i in I)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((quantity_vars[i, j] for i in I)) == d_j[j] for j in J), name='')
    m.addConstrs((quantity_vars[i, j] <= D_j[j] * activation_vars[i] for i in I for j in J), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)