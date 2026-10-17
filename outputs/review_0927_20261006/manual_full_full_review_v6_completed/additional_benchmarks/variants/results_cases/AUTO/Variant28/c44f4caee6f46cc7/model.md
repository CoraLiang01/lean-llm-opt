[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), subject to resource, category, incompatibility, prerequisite, and bundle bonus constraints, using only the items and relationships explicitly listed in the supplied tables.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, logical, and combinatorial constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`I`): All rows in the item table (`item_ref` from table=item).
    - Categories (`C`): All rows in the category table (`category` from table=category).
    - Resources (`R`): All resources in the usage and capacity_ledger tables.
    - Incompatible Pairs (`P`): All pairs from table=incompatible.
    - Prerequisite Pairs (`Q`): All pairs from table=requires.
    - Bundles (`B`): All pairs from table=bundle.
4.  **Define Decision Variables:**
    - `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; forced 0 for unauthorized.
    - `y[i]` = 1 if item `i` is ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `z[c]` = 1 if any item in category `c` is ordered, 0 otherwise. Type: GRB.BINARY.
    - `w[b]` = 1 if both items in bundle `b` are ordered (i.e., both y[item_a[b]] and y[item_b[b]] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - `unit_benefit_cents[i]` (from item table): per-unit benefit for item `i`.
        - `item_fee_cents[i]` (from item table): fixed fee if any of item `i` is ordered.
        - `activation_fee_cents[c]` (from category table): fixed fee if any item in category `c` is ordered.
        - `bonus_cents[b]` (from bundle table): bonus if both items in bundle `b` are ordered.
    - Constraint coefficients:
        - `amount[i,r]` and `unit[i,r]` (from usage table): per-unit resource usage for item `i` and resource `r`.
        - `amount[r]` and `unit[r]` (from capacity_ledger table): signed capacity for resource `r`.
    - Constraint RHS:
        - `minimum_quantity[c]`, `maximum_quantity[c]` (from category table): total quantity bounds per category.
        - `minimum_lot[i]`, `maximum_order[i]` (from item table): per-item order bounds.
        - `authorized[i]` (from item table): eligibility flag for item `i`.
        - Incompatible pairs (`item_a`, `item_b` from incompatible table).
        - Prerequisite pairs (`item_ref`, `prerequisite_ref` from requires table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        sum over items [unit_benefit_cents[i] * x[i] - item_fee_cents[i] * y[i]]
      plus sum over bundles [bonus_cents[b] * w[b]]
      minus sum over categories [activation_fee_cents[c] * z[c]].
7.  **Formulate Constraints:**
    - Resource Capacity: For each resource `r`, sum over items [resource usage per unit (converted to the resource's unit) * x[i]] ≤ total available capacity for `r` (sum of all capacity_ledger entries for `r`, converted to a common unit).
    - Item Authorization and Bounds: For each item `i`, x[i] = 0 if authorized[i] ≠ 1; else, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]]. Enforce: minimum_lot[i] * y[i] ≤ x[i] ≤ maximum_order[i] * y[i], and x[i] = 0 ⇒ y[i] = 0.
    - Category Quantity Bounds: For each category `c`, sum over items in `c` [x[i]] ≥ minimum_quantity[c] and ≤ maximum_quantity[c].
    - Category Activation: For each category `c`, z[c] = 1 if any y[i] = 1 for i in c; else z[c] = 0. Enforce: y[i] ≤ z[c] for all i in c; z[c] ≤ sum over i in c [y[i]].
    - Incompatibility: For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    - Prerequisites: For each (i, prereq), y[i] ≤ y[prereq].
    - Bundle Bonuses: For each bundle b = (i, j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1.
[Abstract Model Plan END]