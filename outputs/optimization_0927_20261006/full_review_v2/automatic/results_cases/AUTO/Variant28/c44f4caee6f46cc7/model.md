[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer order quantities for each authorized product (item) for CENTRAL_FRESH supermarket to maximize net return (in USD cents), considering per-unit benefits, fixed item and category fees, resource capacities, category quantity bounds, incompatibilities, prerequisites, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, logical, and combinatorial constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i ∈ Items): All item_ref rows from the item table with authorized=1.
    - Categories (g ∈ Categories): All category rows from the category table.
    - Resources (r ∈ Resources): All resource rows from the usage and capacity_ledger tables.
    - Bundles (b ∈ Bundles): All bundle rows from the bundle table.
    - Incompatible pairs (p ∈ Pairs): All pairs from the incompatible table.
    - Prerequisite pairs (q ∈ Requires): All pairs from the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer number of cases of item i to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items.
    -   `y[i]` = 1 if any quantity of item i is ordered (i.e., x[i] ≥ minimum_lot[i]), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is ordered (i.e., ∃i in g with x[i] ≥ minimum_lot[i]), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are ordered (i.e., x[item_a[b]] ≥ minimum_lot[item_a[b]] and x[item_b[b]] ≥ minimum_lot[item_b[b]]), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   unit_benefit_cents (from item table, per item)
        -   item_fee_cents (from item table, per item, fixed if any ordered)
        -   activation_fee_cents (from category table, per category, fixed if any item in category ordered)
        -   bonus_cents (from bundle table, per bundle, if both items ordered)
    -   Constraint coefficients:
        -   minimum_lot, maximum_order (from item table, per item)
        -   authorized (from item table, per item)
        -   category (from item table, per item)
        -   minimum_quantity, maximum_quantity (from category table, per category)
        -   resource usage: amount and unit (from usage table, per item-resource)
        -   resource capacity: amount and unit (from capacity_ledger table, per resource)
        -   incompatible pairs (from incompatible table)
        -   prerequisites (from requires table)
    -   Constraint RHS:
        -   Resource capacities: sum of opening and reservation entries per resource (converted to base units)
        -   Category quantity bounds: minimum_quantity and maximum_quantity per category
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: unit_benefit_cents[i] * x[i]
    -   Minus sum over items: item_fee_cents[i] * y[i] (fixed fee if any ordered)
    -   Minus sum over categories: activation_fee_cents[g] * z[g] (fixed fee if any item in category ordered)
    -   Plus sum over bundles: bonus_cents[b] * w[b] (bonus if both items in bundle ordered)
7.  **Formulate Constraints:**
    -   Item order bounds: For each authorized item i, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]]; for unauthorized items, x[i]=0.
    -   Item activation: For each item i, y[i]=1 if x[i] ≥ minimum_lot[i], y[i]=0 if x[i]=0; enforce with x[i] ≥ minimum_lot[i]*y[i] and x[i] ≤ maximum_order[i]*y[i].
    -   Category activation: For each category g, z[g]=1 if any item in g is ordered (i.e., y[i]=1 for some i in g), z[g]=0 otherwise; enforce with y[i] ≤ z[g] for all i in g.
    -   Category quantity bounds: For each category g, sum over i in g of x[i] ≥ minimum_quantity[g] and ≤ maximum_quantity[g].
    -   Resource capacity: For each resource r, sum over items i of (usage amount[i,r] in base units) * x[i] ≤ total available capacity[r] (sum of opening and reservation entries, converted to base units).
    -   Incompatibility: For each incompatible pair (i,j), y[i] + y[j] ≤ 1.
    -   Prerequisites: For each (i, prereq), y[i] ≤ y[prereq].
    -   Bundle bonuses: For each bundle b with items (i,j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1; only bundles with both items authorized can be triggered.
[Abstract Model Plan END]