CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A design team must select exactly one package option from each component family. Each option has a value, '
          'weight, and labor-hour requirement in option_catalog.csv, while total resource limits are listed in '
          'resource_limits.csv.\n'
          '\n'
          'Formulate a maximum-value multi-choice knapsack model. For each family-option pair c-o, define x_co as a '
          'binary variable equal to 1 if option o is selected from family c. The objective is to maximize total '
          'selected value. The model should include exactly-one-option constraints for every family, total weight and '
          'labor-hour constraints, and binary restrictions for all option-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Family', 'Option', 'Value', 'Weight', 'LaborHours'],
             'file_index': 0,
             'file_name': 'option_catalog.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 18,
             'records': [{'source_row': 0,
                          'values': {'Family': 'C1', 'LaborHours': '6', 'Option': 'O1', 'Value': '18', 'Weight': '5'}},
                         {'source_row': 1,
                          'values': {'Family': 'C1', 'LaborHours': '9', 'Option': 'O2', 'Value': '27', 'Weight': '8'}},
                         {'source_row': 2,
                          'values': {'Family': 'C1',
                                     'LaborHours': '11',
                                     'Option': 'O3',
                                     'Value': '30',
                                     'Weight': '10'}},
                         {'source_row': 3,
                          'values': {'Family': 'C2', 'LaborHours': '8', 'Option': 'O1', 'Value': '24', 'Weight': '7'}},
                         {'source_row': 4,
                          'values': {'Family': 'C2',
                                     'LaborHours': '13',
                                     'Option': 'O2',
                                     'Value': '32',
                                     'Weight': '11'}},
                         {'source_row': 5,
                          'values': {'Family': 'C2', 'LaborHours': '10', 'Option': 'O3', 'Value': '28', 'Weight': '9'}},
                         {'source_row': 6,
                          'values': {'Family': 'C3', 'LaborHours': '7', 'Option': 'O1', 'Value': '22', 'Weight': '6'}},
                         {'source_row': 7,
                          'values': {'Family': 'C3',
                                     'LaborHours': '14',
                                     'Option': 'O2',
                                     'Value': '35',
                                     'Weight': '12'}},
                         {'source_row': 8,
                          'values': {'Family': 'C3',
                                     'LaborHours': '12',
                                     'Option': 'O3',
                                     'Value': '31',
                                     'Weight': '10'}},
                         {'source_row': 9,
                          'values': {'Family': 'C4', 'LaborHours': '8', 'Option': 'O1', 'Value': '20', 'Weight': '5'}},
                         {'source_row': 10,
                          'values': {'Family': 'C4', 'LaborHours': '11', 'Option': 'O2', 'Value': '29', 'Weight': '9'}},
                         {'source_row': 11,
                          'values': {'Family': 'C4',
                                     'LaborHours': '13',
                                     'Option': 'O3',
                                     'Value': '34',
                                     'Weight': '12'}},
                         {'source_row': 12,
                          'values': {'Family': 'C5', 'LaborHours': '7', 'Option': 'O1', 'Value': '23', 'Weight': '7'}},
                         {'source_row': 13,
                          'values': {'Family': 'C5',
                                     'LaborHours': '12',
                                     'Option': 'O2',
                                     'Value': '33',
                                     'Weight': '11'}},
                         {'source_row': 14,
                          'values': {'Family': 'C5', 'LaborHours': '10', 'Option': 'O3', 'Value': '26', 'Weight': '8'}},
                         {'source_row': 15,
                          'values': {'Family': 'C6', 'LaborHours': '6', 'Option': 'O1', 'Value': '21', 'Weight': '6'}},
                         {'source_row': 16,
                          'values': {'Family': 'C6',
                                     'LaborHours': '11',
                                     'Option': 'O2',
                                     'Value': '30',
                                     'Weight': '10'}},
                         {'source_row': 17,
                          'values': {'Family': 'C6',
                                     'LaborHours': '15',
                                     'Option': 'O3',
                                     'Value': '36',
                                     'Weight': '13'}}],
             'returned_rows': 18,
             'role': 'option catalog',
             'table_id': 'file_0_view_0'},
            {'columns': ['Resource', 'Limit'],
             'file_index': 1,
             'file_name': 'resource_limits.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Limit': '55', 'Resource': 'Weight'}},
                         {'source_row': 1, 'values': {'Limit': '64', 'Resource': 'LaborHours'}}],
             'returned_rows': 2,
             'role': 'resource limits',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd
import sys

def solve_problem(CSVQA_FRAMES):
    option_frame = CSVQA_FRAMES['file_0_view_0']
    resource_frame = CSVQA_FRAMES['file_1_view_0']
    families = option_frame['Family'].unique().tolist()
    family_options = {}
    for fam in families:
        family_options[fam] = option_frame[option_frame['Family'] == fam]['Option'].unique().tolist()
    co_keys = []
    for fam in families:
        for opt in family_options[fam]:
            co_keys.append((fam, opt))
    v_co = {}
    w_co = {}
    l_co = {}
    for (idx, row) in option_frame.iterrows():
        fam = row['Family']
        opt = row['Option']
        try:
            v = float(row['Value'])
            w = float(row['Weight'])
            l = float(row['LaborHours'])
        except Exception as e:
            raise ValueError(f'Non-numeric value in option_catalog.csv at row {idx}: {e}')
        v_co[fam, opt] = v
        w_co[fam, opt] = w
        l_co[fam, opt] = l
    W_max = None
    L_max = None
    for (idx, row) in resource_frame.iterrows():
        res = row['Resource']
        try:
            lim = float(row['Limit'])
        except Exception as e:
            raise ValueError(f'Non-numeric value in resource_limits.csv at row {idx}: {e}')
        if res == 'Weight':
            W_max = lim
        elif res == 'LaborHours':
            L_max = lim
    if W_max is None or L_max is None:
        raise ValueError('Missing resource limit for Weight or LaborHours in resource_limits.csv')
    for key in co_keys:
        if key not in v_co or key not in w_co or key not in l_co:
            raise ValueError(f'Missing coefficients for (Family, Option) = {key}')
    m = gp.Model('MultiChoiceKnapsack')
    x_vars = m.addVars(co_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((v_co[key] * x_vars[key] for key in co_keys)), gp.GRB.MAXIMIZE)
    for fam in families:
        m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1, name=f'one_option_{fam}')
    m.addConstr(gp.quicksum((w_co[key] * x_vars[key] for key in co_keys)) <= W_max, name='weight_limit')
    m.addConstr(gp.quicksum((l_co[key] * x_vars[key] for key in co_keys)) <= L_max, name='labor_limit')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)