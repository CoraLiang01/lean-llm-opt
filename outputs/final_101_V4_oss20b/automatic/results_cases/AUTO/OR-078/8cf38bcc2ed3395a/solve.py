import gurobipy as gp
import pandas as pd
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
df = pd.read_csv(data_path)
options = df['option'].tolist()
gen_per_lot = dict(zip(df['option'], df['gen_per_lot']))
cost_per_lot = dict(zip(df['option'], df['cost_per_lot']))
demand = 200
m = gp.Model('LotSizing')
x = m.addVars(options, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((cost_per_lot[l] * x[l] for l in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[l] * x[l] for l in options)) >= demand)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('--- Lot purchase plan ---')
    for l in sorted(options):
        qty = x[l].X
        if qty > 0:
            print(f'{l}: {int(qty)} lot(s) -> {gen_per_lot[l] * qty} units')
else:
    print(f'No optimal solution found. Status: {m.status}')