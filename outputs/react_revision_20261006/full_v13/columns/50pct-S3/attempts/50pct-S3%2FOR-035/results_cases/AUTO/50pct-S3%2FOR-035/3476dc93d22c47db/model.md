### Optimization Model

**Sets:**
- $F$: Set of all foods in cost.csv (see Data Mapping for enumeration).

**Parameters (from cost.csv, indexed by $f \in F$):**
- $cal_f$: Calories per serving of food $f$ (column "Calories")
- $prot_f$: Protein (g) per serving of food $f$ (column "Protein(g)")
- $fat_f$: Fat (g) per serving of food $f$ (column "Fat(g)")
- $vitc_f$: Vitamin C (mg) per serving of food $f$ (column "VitaminC(mg)")
- $cost_f$: Cost (USD) per serving of food $f$ (column "Cost")

**Decision Variables:**
- $x_f \geq 0$: Number of servings of food $f$ to include in the meal plan (continuous, may be fractional)

**Objective:**
\[
\min \sum_{f \in F} cost_f \, x_f
\]

**Subject to:**
\[
\sum_{f \in F} cal_f \, x_f \geq 2000 \qquad \text{(Calories requirement)}
\]
\[
\sum_{f \in F} prot_f \, x_f \geq 50 \qquad \text{(Protein requirement)}
\]
\[
\sum_{f \in F} vitc_f \, x_f \geq 60 \qquad \text{(Vitamin C requirement)}
\]
\[
\sum_{f \in F} fat_f \, x_f \leq 70 \qquad \text{(Fat upper bound)}
\]
\[
x_f \geq 0 \qquad \forall f \in F
\]

---

**Data Mapping**

- $F$: All foods in cost.csv, column "Food", table_id: file_0_view_0
- $cal_f$: "Calories", file_0_view_0
- $prot_f$: "Protein(g)", file_0_view_0
- $fat_f$: "Fat(g)", file_0_view_0
- $vitc_f$: "VitaminC(mg)", file_0_view_0
- $cost_f$: "Cost", file_0_view_0
- $x_f$: servings of food $f$ (decision variable, continuous, $\geq 0$)

**Nutritional requirements:**
- Calories: $\geq 2000$ kcal
- Protein: $\geq 50$ g
- Vitamin C: $\geq 60$ mg
- Fat: $\leq 70$ g

**Portions may be fractional.**