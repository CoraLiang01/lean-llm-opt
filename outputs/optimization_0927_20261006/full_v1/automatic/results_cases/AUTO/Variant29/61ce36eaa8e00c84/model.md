[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (total unit benefits plus bundle bonuses, minus item and category activation fees), subject to storage, staff time, and energy limits, category quantity bounds, incompatibility, prerequisite, and bundle bonus rules.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All rows in the 'item' table with `authorized=1`.
    - Categories (`g`): All rows in the 'category' table.
    - Resources (`r`): All unique resources in the 'usage' and 'capacity_ledger' tables.
    - Incompatible pairs (`(i,j)`): All rows in the 'incompatible' table.
    - Prerequisite pairs (`(i,k)`): All rows in the 'requires' table.
    - Bundles (`(i,j)`): All rows in the 'bundle' table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item `i` (authorized items only). Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item `i` is ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `b[i,j]` = 1 if both items `i` and `j` in a bundle are ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` from 'item' table.
        -   `item_fee_cents[i]` from 'item' table.
        -   `activation_fee_cents[g]` from 'category' table.
        -   `bonus_cents[i,j]` from 'bundle' table.
    -   Constraint coefficients:
        -   `amount[i,r]` and `unit[i,r]` from 'usage' table (per-unit resource use).
        -   Resource capacities: sum of `amount[r]` from 'capacity_ledger' table, converted to consistent units.
        -   `minimum_lot[i]`, `maximum_order[i]` from 'item' table.
        -   `minimum_quantity[g]`, `maximum_quantity[g]` from 'category' table.
        -   Incompatibility pairs from 'incompatible' table.
        -   Prerequisite pairs from 'requires' table.
    -   All units (ml, liter, wh, kwh, minute, hour) must be converted to base units (ml, wh, minute) for comparison.
6.  **Formulate Objective:** Maximize total net benefit in cents:
        - Sum over items: `unit_benefit_cents[i] * x[i]`
        - Plus sum over bundles: `bonus_cents[i,j] * b[i,j]`
        - Minus sum over items: `item_fee_cents[i] * y[i]`
        - Minus sum over categories: `activation_fee_cents[g] * z[g]`
7.  **Formulate Constraints:**
    -   Resource Limits: For each resource `r`, sum over items of (per-unit usage of `r` for item `i` * `x[i]`) ≤ total available capacity for `r` (from 'capacity_ledger'), after unit conversion.
    -   Item Quantity Bounds: For each item `i`, `x[i]` ∈ {0} ∪ [minimum_lot[i], maximum_order[i]]; enforce `y[i]=1` if `x[i]>0`, `y[i]=0` if `x[i]=0`.
    -   Category Quantity Bounds: For each category `g`, sum over items in `g` of `x[i]` ≥ `minimum_quantity[g]` and ≤ `maximum_quantity[g]`; set `z[g]=1` if any `x[i]>0` in `g`, else `z[g]=0`.
    -   Incompatibility: For each incompatible pair `(i,j)`, at most one of `y[i]`, `y[j]` can be 1: `y[i] + y[j] ≤ 1`.
    -   Prerequisites: For each prerequisite pair `(i,k)`, `y[i] ≤ y[k]` (if item `i` is ordered, its prerequisite `k` must also be ordered).
    -   Bundle Bonuses: For each bundle `(i,j)`, `b[i,j] ≤ y[i]`, `b[i,j] ≤ y[j]`, and `b[i,j] ≥ y[i] + y[j] - 1` (bonus only if both items are ordered in positive quantity and both are authorized).
    -   Only authorized items may be ordered: restrict all variables and constraints to items with `authorized=1`.
[Abstract Model Plan END]