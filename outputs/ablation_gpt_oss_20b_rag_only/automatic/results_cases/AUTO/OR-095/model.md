**Mathematical Model**

```
Sets
    I   : = { i | i ∈ product_resources.Product }          // Widget1 … Widget141

Parameters
    Labor_i      : = product_resources.LaborHours[i]        // labor hours per unit
    MaterialA_i  : = product_resources.MaterialA[i]        // kg of Material A per unit
    MaterialB_i  : = product_resources.MaterialB[i]        // kg of Material B per unit
    Profit_i     : = product_resources.Profit[i]           // base profit per unit

    LaborLimit   : = resource_limits.MonthlyLimit
                     where Resource = 'LaborHours'           // 5000
    MaterialALimit : = resource_limits.MonthlyLimit
                     where Resource = 'MaterialA'           // 24000
    MaterialBLimit : = resource_limits.MonthlyLimit
                     where Resource = 'MaterialB'           // 15000

    CatalystX_SaleCap : = 1500                               // kg/month
    CatalystX_GenRate : = 5                                  // kg per unit of Widget3

Variables
    x_i   ≥ 0   ∀ i ∈ I          // production quantity of widget i
    s     ≥ 0                 // kg of CatalystX sold

Constraints
    // Resource limits
    Σ_{i∈I} Labor_i   * x_i   ≤ LaborLimit
    Σ_{i∈I} MaterialA_i * x_i ≤ MaterialALimit
    Σ_{i∈I} MaterialB_i * x_i ≤ MaterialBLimit

    // CatalystX handling
    s ≤ CatalystX_SaleCap
    s ≤ CatalystX_GenRate * x_3          // x_3 is the variable for Widget3
    CatalystX_GenRate * x_3 - s ≥ 0      // unsold CatalystX non‑negative

Objective
    maximize  Σ_{i∈I} Profit_i * x_i  +  500 * s  –  1000 * x_3
```

*Explanation of the objective:*  
Revenue from CatalystX sold = 300 $ /kg × s.  
Disposal cost for unsold CatalystX = 200 $ /kg × (5 x₃ – s).  
Thus net CatalystX contribution = 300 s – 200(5 x₃ – s) = 500 s – 1000 x₃.

---

**Data Mapping Rules**

| Symbol | Source Table | Column | Filter | Notes |
|--------|--------------|--------|--------|-------|
| `I` | `file_1_view_0` | `Product` | none | All 141 widgets |
| `Labor_i` | `file_1_view_0` | `LaborHours` | index by `Product` | |
| `MaterialA_i` | `file_1_view_0` | `MaterialA` | index by `Product` | |
| `MaterialB_i` | `file_1_view_0` | `MaterialB` | index by `Product` | |
| `Profit_i` | `file_1_view_0` | `Profit` | index by `Product` | |
| `LaborLimit` | `file_0_view_0` | `MonthlyLimit` | `Resource = 'LaborHours'` | 5000 |
| `MaterialALimit` | `file_0_view_0` | `MonthlyLimit` | `Resource = 'MaterialA'` | 24000 |
| `MaterialBLimit` | `file_0_view_0` | `MonthlyLimit` | `Resource = 'MaterialB'` | 15000 |
| `CatalystX_SaleCap` | constant | – | – | 1500 |
| `CatalystX_GenRate` | constant | – | – | 5 |
| `x_3` | variable | – | corresponds to `Product = 'Widget3'` | |

All other indices and parameters are derived directly from the complete runtime records; no additional rows or columns are introduced. The model respects the original objective, all resource limits, and the CatalystX production‑sale‑disposal logic as specified.