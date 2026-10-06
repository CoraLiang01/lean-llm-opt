#### Abstract Mathematical Model

**Index Set:**
- $F$: set of foods (indexed by $f$), from column `Food` in `file_0_view_0`.

**Parameters:**
- $cal_f$: Calories per serving of food $f$ (`Calories`)
- $prot_f$: Protein (g) per serving of food $f$ (`Protein(g)`)
- $fat_f$: Fat (g) per serving of food $f$ (`Fat(g)`)
- $vitc_f$: Vitamin C (mg) per serving of food $f$ (`VitaminC(mg)`)
- $cost_f$: Cost (USD) per serving of food $f$ (`Cost`)

**Decision Variables:**
- $x_f \geq 0$: Number of servings of food $f$ to include in the meal plan (continuous, may be fractional)

**Objective:**
\[
\min \sum_{f \in F} cost_f \cdot x_f
\]

**Constraints:**
1. **Calorie Requirement:**
   \[
   \sum_{f \in F} cal_f \cdot x_f \geq 2000
   \]
2. **Protein Requirement:**
   \[
   \sum_{f \in F} prot_f \cdot x_f \geq 50
   \]
3. **Vitamin C Requirement:**
   \[
   \sum_{f \in F} vitc_f \cdot x_f \geq 60
   \]
4. **Fat Limit:**
   \[
   \sum_{f \in F} fat_f \cdot x_f \leq 70
   \]
5. **Nonnegativity:**
   \[
   x_f \geq 0 \quad \forall f \in F
   \]

---

#### Data Mapping

- $F$: All values in `Food` column of `file_0_view_0` (cost.csv), source order.
- $cal_f$: `Calories` column, matched to $f$ by `Food`, from `file_0_view_0`.
- $prot_f$: `Protein(g)` column, matched to $f$ by `Food`, from `file_0_view_0`.
- $fat_f$: `Fat(g)` column, matched to $f$ by `Food`, from `file_0_view_0`.
- $vitc_f$: `VitaminC(mg)` column, matched to $f$ by `Food`, from `file_0_view_0`.
- $cost_f$: `Cost` column, matched to $f$ by `Food`, from `file_0_view_0`.