import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others4/cost.csv'
df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
if 'Food' not in df.columns:
    raise KeyError("Required column 'Food' not found in cost.csv")
foods = df['Food'].tolist()

def get_numeric_series(df, colname, entity_ids, entity_label):
    if colname not in df.columns:
        raise KeyError(f"Required column '{colname}' not found in cost.csv")
    s = df.set_index('Food')[colname]
    try:
        vals = s.astype(float)
    except Exception as e:
        raise ValueError(f"Could not convert column '{colname}' to float: {e}")
    missing = [f for f in entity_ids if f not in vals.index]
    if missing:
        raise KeyError(f'Missing {entity_label} values for foods: {missing}')
    return {f: vals[f] for f in entity_ids}
cost = get_numeric_series(df, 'Cost', foods, 'cost')
calories = get_numeric_series(df, 'Calories', foods, 'calories')
protein = get_numeric_series(df, 'Protein(g)', foods, 'protein')
fat = get_numeric_series(df, 'Fat(g)', foods, 'fat')
vitamin_c = get_numeric_series(df, 'VitaminC(mg)', foods, 'vitamin C')
m = gp.Model('MealPlanMinCost')
servings_vars = m.addVars(foods, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[f] * servings_vars[f] for f in foods)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((calories[f] * servings_vars[f] for f in foods)) >= 2000, name='calories_min')
m.addConstr(gp.quicksum((protein[f] * servings_vars[f] for f in foods)) >= 50, name='protein_min')
m.addConstr(gp.quicksum((vitamin_c[f] * servings_vars[f] for f in foods)) >= 60, name='vitaminC_min')
m.addConstr(gp.quicksum((fat[f] * servings_vars[f] for f in foods)) <= 70, name='fat_max')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} USD')
    print('--- Optimal Meal Plan (servings per food) ---')
    for f in foods:
        x = servings_vars[f].X
        if x > 1e-05:
            print(f'{f}: {x:.4f} servings')
    total_cal = sum((calories[f] * servings_vars[f].X for f in foods))
    total_prot = sum((protein[f] * servings_vars[f].X for f in foods))
    total_fat = sum((fat[f] * servings_vars[f].X for f in foods))
    total_vitc = sum((vitamin_c[f] * servings_vars[f].X for f in foods))
    print('\n--- Nutrient Totals ---')
    print(f'Calories: {total_cal:.2f} kcal')
    print(f'Protein: {total_prot:.2f} g')
    print(f'Fat: {total_fat:.2f} g')
    print(f'Vitamin C: {total_vitc:.2f} mg')
else:
    print(f'No optimal solution found. Status: {m.status}')