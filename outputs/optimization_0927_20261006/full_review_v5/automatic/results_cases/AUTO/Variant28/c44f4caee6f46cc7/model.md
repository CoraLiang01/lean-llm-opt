[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), subject to resource, category, incompatibility, prerequisite, and bundle bonus constraints, using the supplied tables as the complete set of options and parameters.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, logical, and combinatorial constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All rows in the item table (`item_ref` from table=item).
    - Categories (`g`): All rows in the category table (`category` from table=category).
    - Resources (`r`): All unique resources in the usage and capacity_ledger tables.
    - Incompatible pairs (`(i,j)`): All pairs from table=incompatible.
    - Prerequisite pairs (`(i,p)`): All pairs from table=requires.
    - Bundles (`(i,j)`): All pairs from table=bundle.
4.  **Define Decision Variables:**
    - `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; forced 0 for unauthorized.
    - `y[i]` = 1 if item `i` is ordered in any positive quantity, 0 otherwise. Type: GRB.BINARY.
    - `z[g]` = 1 if any item in category `g` is ordered, 0 otherwise. Type: GRB.BINARY.
    - `w[i,j]` = 1 if both items `i` and `j` are ordered (for each bundle), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - `unit_benefit_cents[i]` (from table=item): per-unit benefit for item `i`.
        - `item_fee_cents[i]` (from table=item): fixed fee if any of item `i` is ordered.
        - `activation_fee_cents[g]` (from table=category): fixed fee if any item in category `g` is ordered.
        - `bonus_cents[i,j]` (from table=bundle): bonus if both items `i` and `j` are ordered.
    - Constraint coefficients:
        - `amount[i,r]` and `unit[i,r]` (from table=usage): per-unit resource usage for item `i` and resource `r`.
        - `amount[r]` and `unit[r]` (from table=capacity_ledger): signed available capacity for resource `r`.
    - Constraint RHS:
        - `minimum_quantity[g]`, `maximum_quantity[g]` (from table=category): category-level aggregate quantity bounds.
        - `minimum_lot[i]`, `maximum_order[i]` (from table=item): item-level order bounds.
        - Authorization: `authorized[i]` (from table=item): only items with authorized=1 may be ordered.
        - Incompatibility: pairs from table=incompatible.
        - Prerequisites: pairs from table=requires.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        sum over items [unit_benefit_cents[i] * x[i] - item_fee_cents[i] * y[i]]
      plus sum over bundles [bonus_cents[i,j] * w[i,j]]
      minus sum over categories [activation_fee_cents[g] * z[g]].
7.  **Formulate Constraints:**
    - Resource Capacity: For each resource `r`, sum over items [converted usage per unit * x[i]] ≤ sum of all capacity_ledger amounts for resource `r` (convert all units to base units: 1000 ml = 1 liter, 60 minutes = 1 hour, 1000 wh = 1 kwh).
    - Item Authorization and Bounds: For each item `i`, x[i] = 0 if authorized[i] ≠ 1; else, x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i].
    - Item Selection Linking: For each item `i`, y[i] = 1 iff x[i] ≥ 1; enforce y[i] ≤ x[i] / minimum_lot[i] and x[i] ≤ maximum_order[i] * y[i].
    - Category Aggregates: For each category `g`, sum over items in `g` [x[i]] ≥ minimum_quantity[g] and ≤ maximum_quantity[g].
    - Category Activation: For each category `g`, z[g] = 1 iff any item in `g` is ordered; enforce y[i] ≤ z[g] for all items `i` in `g`, and z[g] ≤ sum over i in g [y[i]].
    - Incompatibility: For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    - Prerequisites: For each (i,p), y[i] ≤ y[p].
    - Bundle Bonuses: For each bundle (i,j), w[i,j] ≤ y[i], w[i,j] ≤ y[j], w[i,j] ≥ y[i] + y[j] - 1; only bundles where both items are authorized can earn a bonus.
[Abstract Model Plan END]