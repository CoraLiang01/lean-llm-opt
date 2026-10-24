CSVQA_DATA = {'query': 'A design team must select exactly one package option from each component family. Each option has a value, weight, and labor-hour requirement in option_catalog.csv, while total resource limits are listed in resource_limits.csv.\n\nFormulate a maximum-value multi-choice knapsack model. For each family-option pair c-o, define x_co as a binary variable equal to 1 if option o is selected from family c. The objective is to maximize total selected value. The model should include exactly-one-option constraints for every family, total weight and labor-hour constraints, and binary restrictions for all option-selection variables.', 'relationships': [], 'tables': [{'columns': ['Family', 'Option', 'Value', 'Weight', 'LaborHours'], 'file_name': 'option_catalog.csv', 'records': [{'data_row_number': 0, 'values': {'Family': 'C1', 'LaborHours': '6', 'Option': 'O1', 'Value': '18', 'Weight': '5'}}, {'data_row_number': 1, 'values': {'Family': 'C1', 'LaborHours': '9', 'Option': 'O2', 'Value': '27', 'Weight': '8'}}, {'data_row_number': 2, 'values': {'Family': 'C1', 'LaborHours': '11', 'Option': 'O3', 'Value': '30', 'Weight': '10'}}, {'data_row_number': 3, 'values': {'Family': 'C2', 'LaborHours': '8', 'Option': 'O1', 'Value': '24', 'Weight': '7'}}, {'data_row_number': 4, 'values': {'Family': 'C2', 'LaborHours': '13', 'Option': 'O2', 'Value': '32', 'Weight': '11'}}, {'data_row_number': 5, 'values': {'Family': 'C2', 'LaborHours': '10', 'Option': 'O3', 'Value': '28', 'Weight': '9'}}, {'data_row_number': 6, 'values': {'Family': 'C3', 'LaborHours': '7', 'Option': 'O1', 'Value': '22', 'Weight': '6'}}, {'data_row_number': 7, 'values': {'Family': 'C3', 'LaborHours': '14', 'Option': 'O2', 'Value': '35', 'Weight': '12'}}, {'data_row_number': 8, 'values': {'Family': 'C3', 'LaborHours': '12', 'Option': 'O3', 'Value': '31', 'Weight': '10'}}, {'data_row_number': 9, 'values': {'Family': 'C4', 'LaborHours': '8', 'Option': 'O1', 'Value': '20', 'Weight': '5'}}, {'data_row_number': 10, 'values': {'Family': 'C4', 'LaborHours': '11', 'Option': 'O2', 'Value': '29', 'Weight': '9'}}, {'data_row_number': 11, 'values': {'Family': 'C4', 'LaborHours': '13', 'Option': 'O3', 'Value': '34', 'Weight': '12'}}, {'data_row_number': 12, 'values': {'Family': 'C5', 'LaborHours': '7', 'Option': 'O1', 'Value': '23', 'Weight': '7'}}, {'data_row_number': 13, 'values': {'Family': 'C5', 'LaborHours': '12', 'Option': 'O2', 'Value': '33', 'Weight': '11'}}, {'data_row_number': 14, 'values': {'Family': 'C5', 'LaborHours': '10', 'Option': 'O3', 'Value': '26', 'Weight': '8'}}, {'data_row_number': 15, 'values': {'Family': 'C6', 'LaborHours': '6', 'Option': 'O1', 'Value': '21', 'Weight': '6'}}, {'data_row_number': 16, 'values': {'Family': 'C6', 'LaborHours': '11', 'Option': 'O2', 'Value': '30', 'Weight': '10'}}, {'data_row_number': 17, 'values': {'Family': 'C6', 'LaborHours': '15', 'Option': 'O3', 'Value': '36', 'Weight': '13'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Resource', 'Limit'], 'file_name': 'resource_limits.csv', 'records': [{'data_row_number': 0, 'values': {'Limit': '55', 'Resource': 'Weight'}}, {'data_row_number': 1, 'values': {'Limit': '64', 'Resource': 'LaborHours'}}], 'table_id': 'file_1_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    option_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    option_records = option_table['records']
    limit_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    limit_records = limit_table['records']
    families = set()
    options = set()
    v = {}
    w = {}
    l = {}
    for rec in option_records:
        vals = rec['values']
        c = vals['Family']
        o = vals['Option']
        families.add(c)
        options.add(o)
        v[c, o] = float(vals['Value'])
        w[c, o] = float(vals['Weight'])
        l[c, o] = float(vals['LaborHours'])
    Wmax = None
    Lmax = None
    for rec in limit_records:
        vals = rec['values']
        if vals['Resource'] == 'Weight':
            Wmax = float(vals['Limit'])
        elif vals['Resource'] == 'LaborHours':
            Lmax = float(vals['Limit'])
    if Wmax is None or Lmax is None:
        raise ValueError('Missing resource limits in data')
    for c in families:
        for o in options:
            if (c, o) not in v:
                raise ValueError(f'Missing data for family {c} option {o}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    keys = [(c, o) for c in families for o in options]
    x = m.addVars(keys, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((v[k] * x[k] for k in keys)), GRB.MAXIMIZE)
    for c in families:
        m.addConstr(gurobipy.quicksum((x[c, o] for o in options)) == 1, name=f'One_{c}')
    m.addConstr(gurobipy.quicksum((w[k] * x[k] for k in keys)) <= Wmax, name='WeightLimit')
    m.addConstr(gurobipy.quicksum((l[k] * x[k] for k in keys)) <= Lmax, name='LaborLimit')
    m.optimize()
    return m
m = solve_problem()
