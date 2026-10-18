## Abstract Optimization Model

### Index Set
- \(I\) : Set of food items (indexed by \(i\)).

### Parameters
- \(cal_i\) : Calories per serving of food \(i\).  
- \(prot_i\) : Protein (g) per serving of food \(i\).  
- \(fat_i\) : Fat (g) per serving of food \(i\).  
- \(vitC_i\) : Vitamin C (mg) per serving of food \(i\).  
- \(cost_i\) : Cost (USD) per serving of food \(i\).

### Decision Variables
- \(x_i \ge 0\) : Number of servings of food \(i\) (continuous, fractional servings allowed).

### Objective Function
\[
\min \quad \sum_{i \in I} cost_i \, x_i
\]

### Constraints
1. **Calorie requirement**  
   \[
   \sum_{i \in I} cal_i \, x_i \;\ge\; 2000
   \]

2. **Protein requirement**  
   \[
   \sum_{i \in I} prot_i \, x_i \;\ge\; 50
   \]

3. **Vitamin C requirement**  
   \[
   \sum_{i \in I} vitC_i \, x_i \;\ge\; 60
   \]

4. **Fat limit**  
   \[
   \sum_{i \in I} fat_i \, x_i \;\le\; 70
   \]

5. **Non‑negativity**  
   \[
   x_i \;\ge\; 0 \quad \forall i \in I
   \]

---

## Data Mapping

| Source | Table ID | Columns Used |
|--------|----------|--------------|
| Nutrient & Cost Data | `cost.csv` | `Food`, `Calories`, `Protein(g)`, `Fat(g)`, `VitaminC(mg)`, `Cost` |

- Each parameter \(cal_i, prot_i, fat_i, vitC_i, cost_i\) is mapped directly to the corresponding column in `cost.csv` for food item \(i\).  
- The index set \(I\) corresponds to the distinct values in the `Food` column of `cost.csv`.