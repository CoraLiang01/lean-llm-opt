LEGACY_OBSERVATION = '[{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"A","Demand":"24"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"B","Demand":"18"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"C","Demand":"12"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv","values":{"Item":"D","Demand":"10"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P1","A":"4","B":"0","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P2","A":"0","B":"3","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P3","A":"0","B":"0","C":"2","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P4","A":"0","B":"0","C":"0","D":"2"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P5","A":"2","B":"1","C":"0","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P6","A":"1","B":"0","C":"1","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P7","A":"0","B":"1","C":"0","D":"1"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P8","A":"1","B":"1","C":"1","D":"0"}},{"source":"/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv","values":{"Pattern":"P9","A":"2","B":"0","C":"0","D":"1"}}]'
LEGACY_RECORDS = [{'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'A', 'Demand': '24'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'B', 'Demand': '18'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'C', 'Demand': '12'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/item_demand.csv', 'values': {'Item': 'D', 'Demand': '10'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P1', 'A': '4', 'B': '0', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P2', 'A': '0', 'B': '3', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P3', 'A': '0', 'B': '0', 'C': '2', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P4', 'A': '0', 'B': '0', 'C': '0', 'D': '2'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P5', 'A': '2', 'B': '1', 'C': '0', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P6', 'A': '1', 'B': '0', 'C': '1', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P7', 'A': '0', 'B': '1', 'C': '0', 'D': '1'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P8', 'A': '1', 'B': '1', 'C': '1', 'D': '0'}}, {'source': '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant9/inputs/cutting_patterns.csv', 'values': {'Pattern': 'P9', 'A': '2', 'B': '0', 'C': '0', 'D': '1'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
demand = {}
for rec in records:
    if rec['source'].endswith('item_demand.csv'):
        item = rec['values']['Item']
        demand[item] = int(rec['values']['Demand'])
patterns = []
pattern_yield = {}
for rec in records:
    if rec['source'].endswith('cutting_patterns.csv'):
        p = rec['values']['Pattern']
        patterns.append(p)
        pattern_yield[p] = {}
        for item in ['A', 'B', 'C', 'D']:
            pattern_yield[p][item] = int(rec['values'][item])
items = sorted(demand.keys())
patterns = sorted(patterns, key=lambda x: int(x[1:]))
for p in patterns:
    for i in items:
        if i not in pattern_yield[p]:
            raise ValueError(f'Missing yield for pattern {p}, item {i}')
for i in items:
    if i not in demand:
        raise ValueError(f'Missing demand for item {i}')
m = gp.Model('cutting_stock')
y_vars = m.addVars(patterns, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((y_vars[p] for p in patterns)), GRB.MINIMIZE)
for i in items:
    m.addConstr(gp.quicksum((pattern_yield[p][i] * y_vars[p] for p in patterns)) >= demand[i], name=f'demand_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')