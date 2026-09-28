#### Abstract Optimization Model

**Index Set:**
- $F$: set of all foods (from column `Food` in `cost.csv`)

**Parameters (for each $f \in F$):**
- $cal_f$: Calories per serving of food $f$ (`Calories`)
- $prot_f$: Protein (g) per serving of food $f$ (`Protein(g)`)
- $fat_f$: Fat (g) per serving of food $f$ (`Fat(g)`)
- $vitc_f$: Vitamin C (mg) per serving of food $f$ (`VitaminC(mg)`)
- $cost_f$: Cost (USD) per serving of food $f$ (`Cost`)

**Decision Variables:**
- $x_f \geq 0$: number of servings of food $f$ (continuous, may be fractional)

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
4. **Fat Upper Bound:**
   \[
   \sum_{f \in F} fat_f \cdot x_f \leq 70
   \]
5. **Nonnegativity:**
   \[
   x_f \geq 0 \quad \forall f \in F
   \]

---

**Data Mapping:**

- Table: `cost.csv` (table_id: `file_0_view_0`)
    - Index set $F$: column `Food`
    - $cal_f$: column `Calories`
    - $prot_f$: column `Protein(g)`
    - $fat_f$: column `Fat(g)`
    - $vitc_f$: column `VitaminC(mg)`
    - $cost_f$: column `Cost`