[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), subject to resource, category, incompatibility, prerequisite, and bundle bonus constraints, using the supplied tables as the complete set of options and parameters.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, group activation, incompatibility, and logical (prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All rows in the item table with `authorized=1`.
    - Categories (`g`): All rows in the category table.
    - Resources (`r`): All resources in the usage and capacity_ledger tables.
    - Incompatible pairs (`(i,j)`): All pairs from the incompatible table.
    - Prerequisite pairs (`(i,p)`): All pairs from the requires table.
    - Bundles (`(i,j)`): All pairs from the bundle table.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]].
    -   `y[i]` = 1 if item `i` is ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `w[i,j]` = 1 if both items `i` and `j` in a bundle are ordered (i.e., x[i]>0 and x[j]>0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from item table): per-unit benefit for item `i`.
        -   `item_fee_cents[i]` (from item table): fixed fee if any of item `i` is ordered.
        -   `activation_fee_cents[g]` (from category table): fixed fee if any item in category `g` is ordered.
        -   `bonus_cents[i,j]` (from bundle table): bonus if both items `i` and `j` are ordered.
    -   Constraint coefficients:
        -   `amount[i,r]` and `unit[i,r]` (from usage table): per-unit resource usage for item `i` and resource `r`.
        -   `amount[r]` and `unit[r]` (from capacity_ledger table): signed sum of available capacity for resource `r`.
    -   Constraint RHS:
        -   `minimum_lot[i]`, `maximum_order[i]` (from item table): per-item order bounds.
        -   `minimum_quantity[g]`, `maximum_quantity[g]` (from category table): per-category aggregate order bounds.
        -   Incompatible pairs: from incompatible table.
        -   Prerequisite pairs: from requires table.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (unit_benefit_cents[i] * x[i]) minus (item_fee_cents[i] * y[i])
    -   Minus sum over all categories: (activation_fee_cents[g] * z[g])
    -   Plus sum over all bundles: (bonus_cents[i,j] * w[i,j])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:** For each item `i`, enforce x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i], and y[i] = 1 iff x[i] > 0.
    -   **Category activation and bounds:** For each category `g`, z[g] = 1 iff any y[i]=1 for item `i` in category `g`; sum of x[i] over items in `g` must satisfy minimum_quantity[g] ≤ sum ≤ maximum_quantity[g].
    -   **Resource capacity:** For each resource `r`, sum over items of (amount[i,r] * x[i] converted to the unit of capacity_ledger[r]) ≤ total available capacity for `r` (sum of opening and reservation entries, converted to the same unit).
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   **Prerequisites:** For each requires pair (i,p), y[i] ≤ y[p].
    -   **Bundle bonuses:** For each bundle (i,j), w[i,j] = 1 iff y[i]=1 and y[j]=1; i.e., w[i,j] ≤ y[i], w[i,j] ≤ y[j], w[i,j] ≥ y[i] + y[j] - 1.
[Abstract Model Plan END]