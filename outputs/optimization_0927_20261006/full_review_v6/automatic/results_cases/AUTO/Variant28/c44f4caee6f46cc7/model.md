[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), subject to resource, category, incompatibility, prerequisite, and bundle bonus constraints, using the supplied tables as the complete set of options and parameters.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, logical, and combinatorial constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All item rows from the `item` table (export_06.csv) with `authorized=1`.
    - Categories (`g`): All category rows from the `category` table (export_03.csv).
    - Resources (`r`): All resource types from the `usage` and `capacity_ledger` tables (export_09.csv, export_02.csv).
    - Incompatible pairs (`(i,j)`): All pairs from the `incompatible` table (export_05.csv) where both items are authorized.
    - Prerequisite pairs (`(i,p)`): All pairs from the `requires` table (export_08.csv) where both items are authorized.
    - Bundles (`(i,j)`): All pairs from the `bundle` table (export_01.csv) where both items are authorized.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item `i` is ordered in any positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[i,j]` = 1 if both items `i` and `j` in a bundle are ordered in positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from `item` table): per-unit benefit for item `i`.
        -   `item_fee_cents[i]` (from `item` table): fixed fee if any of item `i` is ordered.
        -   `activation_fee_cents[g]` (from `category` table): fixed fee if any item in category `g` is ordered.
        -   `bonus_cents[i,j]` (from `bundle` table): bonus if both items `i` and `j` are ordered.
    -   Constraint coefficients:
        -   `amount[i,r]` and `unit[i,r]` (from `usage` table): per-unit resource usage for item `i` and resource `r`.
        -   `amount[r]` and `unit[r]` (from `capacity_ledger` table): signed sum of available capacity for resource `r`.
    -   Constraint RHS:
        -   `minimum_quantity[g]`, `maximum_quantity[g]` (from `category` table): total quantity bounds per category.
        -   `minimum_lot[i]`, `maximum_order[i]` (from `item` table): per-item order bounds.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: `unit_benefit_cents[i] * x[i]`
    -   Minus sum over all items: `item_fee_cents[i] * y[i]` (fixed fee if item ordered)
    -   Minus sum over all categories: `activation_fee_cents[g] * z[g]` (fixed fee if any item in category ordered)
    -   Plus sum over all bundles: `bonus_cents[i,j] * b[i,j]` (bonus if both items in bundle ordered)
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item `i`, enforce:
        -   `x[i] = 0` or `minimum_lot[i] ≤ x[i] ≤ maximum_order[i]`
        -   `y[i] = 1` iff `x[i] ≥ minimum_lot[i]`; `y[i] = 0` iff `x[i] = 0`
        -   Implement as: `x[i] ≥ minimum_lot[i] * y[i]`, `x[i] ≤ maximum_order[i] * y[i]`, `x[i] ≥ 0`
    -   **Category activation and quantity bounds:** For each category `g`:
        -   `z[g] = 1` iff any item in category `g` is ordered (`y[i]` for any `i` in `g`)
        -   `z[g] ≥ y[i]` for all `i` in `g`; `z[g] ≤ sum_{i in g} y[i]`
        -   Total quantity in category: `minimum_quantity[g] ≤ sum_{i in g} x[i] ≤ maximum_quantity[g]`
    -   **Resource capacity:** For each resource `r`:
        -   Convert all item usages and capacity ledger entries to a common unit (ml, wh, minute), using 1000 ml/liter, 60 min/hour, 1000 wh/kwh as needed.
        -   Total usage: `sum_{i} (per-unit usage of r by i in base units) * x[i] ≤ total available capacity of r in base units` (sum of all `capacity_ledger` entries for `r`)
    -   **Incompatibility:** For each incompatible pair `(i,j)`:
        -   `y[i] + y[j] ≤ 1` (cannot order both)
    -   **Prerequisite:** For each requires pair `(i,p)`:
        -   `y[i] ≤ y[p]` (if item `i` is ordered, its prerequisite `p` must also be ordered)
    -   **Bundle bonuses:** For each bundle `(i,j)`:
        -   `b[i,j] ≤ y[i]`, `b[i,j] ≤ y[j]`, `b[i,j] ≥ y[i] + y[j] - 1` (bonus only if both items ordered)
        -   Only include bundle if both items are authorized.
[Abstract Model Plan END]