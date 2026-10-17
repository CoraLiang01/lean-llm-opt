**Mathematical Model**

```
Sets
  F  : set of foods  (Food identifiers from the table)

Parameters
  Calories_f   : calories per serving of food f
  Protein_f    : protein (g) per serving of food f
  Fat_f        : fat (g) per serving of food f
  VitaminC_f   : vitamin‑C (mg) per serving of food f
  Cost_f       : cost (USD) per serving of food f

Variables
  x_f ≥ 0      : number of servings of food f  (continuous)

Objective
  minimize   Z = Σ_{f∈F} Cost_f · x_f

Constraints
  Σ_{f∈F} Calories_f · x_f   ≥ 2000          (calories)
  Σ_{f∈F} Protein_f  · x_f   ≥ 50            (protein)
  Σ_{f∈F} VitaminC_f · x_f   ≥ 60            (vitamin‑C)
  Σ_{f∈F} Fat_f      · x_f   ≤ 70            (fat)
```

---

**Data Mapping**

| Parameter | Table ID          | Column Name      |
|-----------|-------------------|------------------|
| Food      | file_0_view_0     | Food             |
| Calories_f| file_0_view_0     | Calories         |
| Protein_f | file_0_view_0     | Protein(g)       |
| Fat_f     | file_0_view_0     | Fat(g)           |
| VitaminC_f| file_0_view_0     | VitaminC(mg)     |
| Cost_f    | file_0_view_0     | Cost             |

The index set `F` is the set of all distinct `Food` values present in the table `file_0_view_0`. All parameters are retrieved from the corresponding columns of that table. All decision variables `x_f` are continuous and non‑negative, allowing fractional servings. The objective is to minimize total cost while satisfying the nutritional constraints.