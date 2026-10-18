#### Index Sets

- $F$: set of all foods (from column "Food" in table_id: file_0_view_0).

#### Parameters

For each $f \in F$ (from table_id: file_0_view_0):

- $cal_f$: Calories per serving of food $f$ (column "Calories")
- $prot_f$: Protein (g) per serving of food $f$ (column "Protein(g)")
- $fat_f$: Fat (g) per serving of food $f$ (column "Fat(g)")
- $vitc_f$: Vitamin C (mg) per serving of food $f$ (column "VitaminC(mg)")
- $cost_f$: Cost (USD) per serving of food $f$ (column "Cost")

#### Decision Variables

- $x_f \geq 0$: Number of servings of food $f$ to include in the meal plan (continuous, can be fractional), for all $f \in F$.

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

- All foods, nutrients, and cost data are from table_id: file_0_view_0, columns: "Food", "Calories", "Protein(g)", "Fat(g)", "VitaminC(mg)", "Cost" in cost.csv, with no filters (all 120 rows used).