#### Sets
- $F$: set of all foods (indexed by $f$).

#### Parameters
- $cal_f$: Calories per serving of food $f$ (from column "Calories" in table_id: file_0_view_0).
- $prot_f$: Protein (g) per serving of food $f$ (from column "Protein(g)" in table_id: file_0_view_0).
- $fat_f$: Fat (g) per serving of food $f$ (from column "Fat(g)" in table_id: file_0_view_0).
- $vitc_f$: Vitamin C (mg) per serving of food $f$ (from column "VitaminC(mg)" in table_id: file_0_view_0).
- $cost_f$: Cost (USD) per serving of food $f$ (from column "Cost" in table_id: file_0_view_0).

#### Decision Variables
- $x_f \geq 0$: Number of servings of food $f$ to include in the meal plan (continuous, may be fractional).

#### Objective
$$
\min \sum_{f \in F} cost_f \cdot x_f
$$

#### Constraints

1. **Calorie Requirement**
   $$
   \sum_{f \in F} cal_f \cdot x_f \geq 2000
   $$

2. **Protein Requirement**
   $$
   \sum_{f \in F} prot_f \cdot x_f \geq 50
   $$

3. **Vitamin C Requirement**
   $$
   \sum_{f \in F} vitc_f \cdot x_f \geq 60
   $$

4. **Fat Upper Bound**
   $$
   \sum_{f \in F} fat_f \cdot x_f \leq 70
   $$

5. **Nonnegativity**
   $$
   x_f \geq 0 \quad \forall f \in F
   $$

---

#### Data Mapping

- All sets and parameters are defined using all records and columns from table_id: file_0_view_0 (cost.csv), with:
    - $F$: all foods from column "Food"
    - $cal_f$: column "Calories"
    - $prot_f$: column "Protein(g)"
    - $fat_f$: column "Fat(g)"
    - $vitc_f$: column "VitaminC(mg)"
    - $cost_f$: column "Cost"
- No filters were applied; all 120 rows are included as returned by CSVQA.