LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"A","Demand":"24"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"B","Demand":"18"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"C","Demand":"12"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"D","Demand":"10"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P1","A":"4","B":"0","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P2","A":"0","B":"3","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P3","A":"0","B":"0","C":"2","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P4","A":"0","B":"0","C":"0","D":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P5","A":"2","B":"1","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P6","A":"1","B":"0","C":"1","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P7","A":"0","B":"1","C":"0","D":"1"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P8","A":"1","B":"1","C":"1","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P9","A":"2","B":"0","C":"0","D":"1"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'A', 'Demand': '24'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'B', 'Demand': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'C', 'Demand': '12'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'D', 'Demand': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P1', 'A': '4', 'B': '0', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P2', 'A': '0', 'B': '3', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P3', 'A': '0', 'B': '0', 'C': '2', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P4', 'A': '0', 'B': '0', 'C': '0', 'D': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P5', 'A': '2', 'B': '1', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P6', 'A': '1', 'B': '0', 'C': '1', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P7', 'A': '0', 'B': '1', 'C': '0', 'D': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P8', 'A': '1', 'B': '1', 'C': '1', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P9', 'A': '2', 'B': '0', 'C': '0', 'D': '1'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
item_demand = {}
for rec in records:
    if rec['source'].endswith('item_demand.csv'):
        item = rec['values']['Item']
        demand = int(rec['values']['Demand'])
        if item in item_demand:
            raise ValueError(f'Duplicate demand for item {item}')
        item_demand[item] = demand
patterns = []
pattern_item_units = {}
for rec in records:
    if rec['source'].endswith('cutting_patterns.csv'):
        pat = rec['values']['Pattern']
        if pat in pattern_item_units:
            raise ValueError(f'Duplicate pattern {pat}')
        patterns.append(pat)
        pattern_item_units[pat] = {}
        for item in item_demand:
            if item not in rec['values']:
                raise ValueError(f'Missing item {item} in pattern {pat}')
            pattern_item_units[pat][item] = int(rec['values'][item])
if not item_demand:
    raise ValueError('No item demand found')
if not patterns:
    raise ValueError('No patterns found')
for pat in patterns:
    for item in item_demand:
        if item not in pattern_item_units[pat]:
            raise ValueError(f'Pattern {pat} missing item {item}')
m = gp.Model('cutting_stock_pattern_selection')
y_vars = m.addVars(patterns, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), GRB.MINIMIZE)
for item in item_demand:
    m.addConstr(gp.quicksum((pattern_item_units[p][item] * y_vars[p] for p in patterns)) >= item_demand[item], name=f'demand_{item}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')