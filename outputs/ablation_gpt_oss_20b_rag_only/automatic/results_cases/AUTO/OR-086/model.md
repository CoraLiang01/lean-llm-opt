**Mathematical model**

```
SETS
    G   : {I, II, III}          ! raw‑grade identifiers
    B   : {Red, Yellow, Blue}   ! wine‑brand identifiers

PARAMETERS
    supply[g]   ≥ 0              ! daily supply limit (kg) – from file_0_view_0
    cost[g]     ≥ 0              ! unit cost (CNY/kg) – from file_0_view_0
    price[b]    ≥ 0              ! selling price (CNY/kg) – from file_1_view_0

    lower[b,g]  ∈ [0,1]          ! lower proportion bound – defined only when
    upper[b,g]  ∈ [0,1]          ! an upper bound is specified in the text

VARIABLES
    x[g,b]      ≥ 0              ! kg of grade g used for brand b

OBJECTIVE
    maximize  Z = Σ_{b∈B} price[b] * Σ_{g∈G} x[g,b]
               - Σ_{g∈G} cost[g]   * Σ_{b∈B} x[g,b]

CONSTRAINTS
    ! Raw‑material supply limits
    supply_con[g] : Σ_{b∈B} x[g,b] ≤ supply[g]          ∀ g∈G

    ! Blending‑proportion constraints
    let T_b = Σ_{g∈G} x[g,b]   ! total production of brand b

    lower_con[b,g] : x[g,b] ≥ lower[b,g] * T_b            ∀ (b,g) with lower[b,g] defined
    upper_con[b,g] : x[g,b] ≤ upper[b,g] * T_b            ∀ (b,g) with upper[b,g] defined

    ! Minimum production for the Red brand
    min_red : T_Red ≥ 2000
```

All variables are continuous and non‑negative.  
The model is a linear program.

---

**Data‑mapping rules**

| Table ID | Source column | Target parameter | Mapping rule |
|----------|---------------|------------------|--------------|
| file_0_view_0 | `Grade` | `g` | Each distinct grade value becomes an element of set G. |
| | `Daily Supply (kg)` | `supply[g]` | `supply[g]` ← value of this column for grade g. |
| | `Cost (CNY/kg)` | `cost[g]` | `cost[g]` ← value of this column for grade g. |
| file_1_view_0 | `Brand` | `b` | Each distinct brand value becomes an element of set B. |
| | `Selling Price (CNY/kg)` | `price[b]` | `price[b]` ← value of this column for brand b. |
| | `Blending Requirements` | `lower[b,g]`, `upper[b,g]` | Parse the free‑text string for each brand b:<br>• Clause “Grade less than X%” → `upper[b,Grade] = X/100`.<br>• Clause “Grade more than Y%” → `lower[b,Grade] = Y/100`. <br>Only grades that appear in the clause receive a bound; all other `(b,g)` pairs have no bound. |

The minimum‑production requirement `T_Red ≥ 2000` is hard‑coded in the model and does not come from the data tables.

No other parameters or constraints are introduced. All indices in constraints belong to the declared sets, and the model respects the data supplied in the two CSV files.