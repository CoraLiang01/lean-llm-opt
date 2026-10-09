[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), subject to resource, category, incompatibility, prerequisite, and bundle bonus constraints, using the supplied tables as the complete set of options and parameters.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, group activation, incompatibility, prerequisite, and bundle logic.
3.  **Define Index Sets:** The primary indices are:
    - Items (`i`): All rows in the item table (`item_ref` from table=item).
    - Categories (`g`): All rows in the category table (`category` from table=category).
    - Resources (`r`): All rows in the resource usage and capacity tables (`resource` from table=usage and table=capacity_ledger).
    - Incompatible pairs (`(i,j)`): All rows in the incompatible table (`item_a`, `item_b` from table=incompatible).
    - Prerequisite pairs (`(i,p)`): All rows in the requires table (`item_ref`, `prerequisite_ref` from table=requires).
    - Bundles (`(i,j)`): All rows in the bundle table (`item_a`, `item_b` from table=bundle).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of cases of item `i` to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items; forced 0 for unauthorized.
    -   `y[i]` = 1 if item `i` is ordered in any positive quantity, 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is ordered, 0 otherwise. Type: GRB.BINARY.
    -   `b[i,j]` = 1 if both items `i` and `j` are ordered in positive quantity (for bundle bonuses), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   `unit_benefit_cents[i]` (from table=item, column=unit_benefit_cents).
        -   `item_fee_cents[i]` (from table=item, column=item_fee_cents).
        -   `activation_fee_cents[g]` (from table=category, column=activation_fee_cents).
        -   `bonus_cents[i,j]` (from table=bundle, column=bonus_cents).
    -   Constraint coefficients:
        -   `minimum_lot[i]`, `maximum_order[i]` (from table=item).
        -   `authorized[i]` (from table=item).
        -   `category[i]` (from table=item, column=category).
        -   `minimum_quantity[g]`, `maximum_quantity[g]` (from table=category).
        -   `usage[i,r]` (from table=usage, columns=item_ref, resource, amount, unit).
        -   `capacity_ledger[r]` (from table=capacity_ledger, sum of amount per resource, after unit conversion).
        -   Incompatibility pairs (`item_a`, `item_b` from table=incompatible).
        -   Prerequisite pairs (`item_ref`, `prerequisite_ref` from table=requires).
    -   Constraint RHS:
        -   Resource capacities: sum of opening and reservation entries per resource, after converting all units to base units (ml, wh, minute).
        -   Category quantity bounds: minimum_quantity[g], maximum_quantity[g].
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        -   Sum over all items: (unit_benefit_cents[i] * x[i]) 
        -   Minus sum over all items: (item_fee_cents[i] * y[i]) [fee paid once per item if any ordered]
        -   Minus sum over all categories: (activation_fee_cents[g] * z[g]) [fee paid once per category if any item in group ordered]
        -   Plus sum over all bundles: (bonus_cents[i,j] * b[i,j]) [bonus earned once per bundle if both items ordered]
7.  **Formulate Constraints:**
    -   **Item selection and bounds:**
        -   For each item `i`: 
            -   If authorized[i]=1: x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (enforced via y[i] and bounds: minimum_lot[i]*y[i] ≤ x[i] ≤ maximum_order[i]*y[i]; x[i]=0 iff y[i]=0).
            -   If authorized[i]=0: x[i]=0, y[i]=0.
    -   **Category activation and quantity bounds:**
        -   For each category `g`: 
            -   z[g] ≥ y[i] for all items i in category g; z[g] ≤ sum over i in g of y[i].
            -   sum over i in g of x[i] ≥ minimum_quantity[g].
            -   sum over i in g of x[i] ≤ maximum_quantity[g].
    -   **Resource capacity constraints:**
        -   For each resource `r`: 
            -   sum over items i of (usage[i,r] * x[i]) ≤ total available capacity for r (sum of capacity_ledger[r], after converting all usage and capacity to base units: 1000 ml/liter, 1000 wh/kwh, 60 min/hour).
    -   **Incompatibility constraints:**
        -   For each incompatible pair (i,j): y[i] + y[j] ≤ 1.
    -   **Prerequisite constraints:**
        -   For each requires pair (i,p): y[i] ≤ y[p] (if i is ordered, its prerequisite p must also be ordered).
    -   **Bundle bonus logic:**
        -   For each bundle (i,j): 
            -   b[i,j] ≤ y[i], b[i,j] ≤ y[j], b[i,j] ≥ y[i] + y[j] - 1.
            -   Only bundles where both i and j are authorized can be triggered (if either is unauthorized, b[i,j]=0).
    -   **Variable domains:**
        -   x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer).
        -   y[i], z[g], b[i,j] ∈ {0,1} (binary).
[Abstract Model Plan END]