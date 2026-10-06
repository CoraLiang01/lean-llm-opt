#### Abstract Mathematical Model

**Index Set:**
- $F$ = set of foods (from `cost.csv`, column `Food`)

**Parameters:**
- $c_f$ = cost per serving of food $f$ (`cost.csv`, column `Cost`)
- $cal_f$ = calories per serving of food $f$ (`cost.csv`, column `Calories`)
- $prot_f$ = protein (g) per serving of food $f$ (`cost.csv`, column `Protein(g)`)
- $fat_f$ = fat (g) per serving of food $f$ (`cost.csv`, column `Fat(g)`)
- $vitc_f$ = vitamin C (mg) per serving of food $f$ (`cost.csv`, column `VitaminC(mg)`)

**Decision Variables:**
- $x_f \geq 0$ = number of servings of food $f$ to include in the meal plan (continuous, may be fractional)

**Objective:**
\[
\min \sum_{f \in F} c_f x_f
\]

**Subject to:**
\[
\sum_{f \in F} cal_f \, x_f \geq 2000
\]
\[
\sum_{f \in F} prot_f \, x_f \geq 50
\]
\[
\sum_{f \in F} vitc_f \, x_f \geq 60
\]
\[
\sum_{f \in F} fat_f \, x_f \leq 70
\]
\[
x_f \geq 0 \quad \forall f \in F
\]

---

#### Data Mapping

| Symbol      | Source Table/Column                | Description                                 |
|-------------|------------------------------------|---------------------------------------------|
| $F$         | `file_0_view_0`, `Food`            | Set of foods                                |
| $c_f$       | `file_0_view_0`, `Cost`            | Cost per serving of food $f$                |
| $cal_f$     | `file_0_view_0`, `Calories`        | Calories per serving of food $f$            |
| $prot_f$    | `file_0_view_0`, `Protein(g)`      | Protein (g) per serving of food $f$         |
| $fat_f$     | `file_0_view_0`, `Fat(g)`          | Fat (g) per serving of food $f$             |
| $vitc_f$    | `file_0_view_0`, `VitaminC(mg)`    | Vitamin C (mg) per serving of food $f$      |
| $x_f$       | (decision variable)                | Servings of food $f$ (continuous, $\geq 0$) |

---

**All foods and their nutrient/cost values are taken directly from `cost.csv` as indexed above.**