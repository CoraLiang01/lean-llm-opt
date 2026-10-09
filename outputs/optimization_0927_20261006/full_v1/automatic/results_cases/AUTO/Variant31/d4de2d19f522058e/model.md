[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints, using all supplied table rows directly.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, category, compatibility, and logical (dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i ∈ Items): from all rows in the item table.
    - Categories (g ∈ Categories): from all rows in the category table.
    - Resources (r ∈ Resources): from all rows in the capacity_ledger and usage tables.
    - Bundles (b ∈ Bundles): from all rows in the bundle table.
    - Incompatible pairs (p ∈ Incompatibles): from all rows in the incompatible table.
    - Requires pairs (q ∈ Requires): from all rows in the requires table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity of item i to order (must be 0 if unauthorized; otherwise between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `y[i]` = 1 if any quantity of item i is ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is ordered (i.e., sum over i in g of y[i] ≥ 1), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are ordered (i.e., y[item_a] = 1 and y[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   unit_benefit_cents (from item table, per item)
        -   item_fee_cents (from item table, per item, fixed if x[i] > 0)
        -   activation_fee_cents (from category table, per category, fixed if any item in category is ordered)
        -   bonus_cents (from bundle table, per bundle, awarded if both items in bundle are ordered)
    -   Constraint coefficients:
        -   usage amount (from usage table: amount of resource r used per unit of item i)
        -   capacity_ledger amount (from capacity_ledger table: total available for each resource r)
        -   minimum_lot, maximum_order (from item table, per item)
        -   authorized (from item table, per item)
        -   minimum_quantity, maximum_quantity (from category table, per category)
        -   incompatible pairs (from incompatible table: pairs of items that cannot both be ordered)
        -   requires pairs (from requires table: item i requires prerequisite item j to be ordered if i is ordered)
    -   Constraint RHS:
        -   Resource limits (from capacity_ledger: sum of opening and reservation for each resource)
        -   Category quantity limits (from category table: minimum_quantity and maximum_quantity)
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items of (unit_benefit_cents[i] * x[i]) 
    -   Minus sum over all items of (item_fee_cents[i] * y[i]) [fixed charge per item if any ordered]
    -   Minus sum over all categories of (activation_fee_cents[g] * z[g]) [fixed charge per category if any item in category is ordered]
    -   Plus sum over all bundles of (bonus_cents[b] * w[b]) [bonus if both items in bundle are ordered]
7.  **Formulate Constraints:**
    -   Resource Capacity: For each resource r, sum over all items of (usage amount of r per unit of i * x[i]) ≤ total available capacity_ledger amount for r (sum of opening and reservation).
    -   Item Authorization and Lot Size: For each item i, x[i] = 0 if authorized[i] = 0; if authorized[i] = 1, then minimum_lot[i] ≤ x[i] ≤ maximum_order[i] or x[i] = 0.
    -   Category Quantity Limits: For each category g, sum over all items in g of x[i] ≥ minimum_quantity[g] and ≤ maximum_quantity[g] (unconditional, even if no item is selected).
    -   Category Activation: For each category g, z[g] = 1 if any y[i] = 1 for i in g; z[g] = 0 otherwise.
    -   Item Activation: For each item i, y[i] = 1 if x[i] > 0; y[i] = 0 if x[i] = 0.
    -   Bundle Bonus: For each bundle b, w[b] = 1 if both y[item_a] = 1 and y[item_b] = 1; w[b] = 0 otherwise.
    -   Incompatibility: For each incompatible pair (i, j), y[i] + y[j] ≤ 1 (cannot both be selected).
    -   Requires Dependency: For each requires pair (i, j), x[i] > 0 ⇒ x[j] > 0 (if i is ordered, prerequisite j must also be ordered in positive quantity).
[Abstract Model Plan END]