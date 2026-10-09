[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer number of cases to order for each authorized product for the CENTRAL_FRESH supermarket, maximizing net return (total benefit minus all fixed and variable fees), while respecting resource capacities, item/category bounds, incompatibilities, prerequisites, and bundle bonuses.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility/prerequisite) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i ∈ Items): All item rows in the item table (export_06.csv) with authorized=1.
    - Categories (g ∈ Categories): All category rows in the category table (export_03.csv).
    - Resources (r ∈ Resources): All resource types in the usage and capacity_ledger tables (export_09.csv, export_02.csv).
    - Bundles (b ∈ Bundles): All bundle rows in the bundle table (export_01.csv).
    - Incompatible pairs (p ∈ Incompatibles): All pairs in the incompatible table (export_05.csv).
    - Prerequisite pairs (q ∈ Prerequisites): All pairs in the requires table (export_08.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of cases of item i to order. Type: GRB.INTEGER, domain: {0} ∪ [minimum_lot[i], maximum_order[i]] for authorized items.
    -   `y[i]` = 1 if any quantity of item i is ordered (i.e., x[i] ≥ minimum_lot[i]), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category g is ordered (i.e., ∃i in g with x[i] ≥ minimum_lot[i]), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are ordered with positive quantity, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        -   unit_benefit_cents[i] (from item table, export_06.csv)
        -   item_fee_cents[i] (from item table, export_06.csv)
        -   activation_fee_cents[g] (from category table, export_03.csv)
        -   bonus_cents[b] (from bundle table, export_01.csv)
    -   Constraint coefficients:
        -   minimum_lot[i], maximum_order[i] (from item table)
        -   minimum_quantity[g], maximum_quantity[g] (from category table)
        -   usage: amount[i, r], unit[i, r] (from usage table, export_09.csv)
        -   capacity_ledger: amount[r, e], unit[r, e] (from capacity_ledger table, export_02.csv)
        -   Incompatibles: item_a, item_b (from incompatible table, export_05.csv)
        -   Prerequisites: item_ref, prerequisite_ref (from requires table, export_08.csv)
    -   Constraint RHS:
        -   Resource capacities: sum of opening and reservation entries per resource (converted to base units: ml, wh, minute)
        -   Category quantity bounds: minimum_quantity[g], maximum_quantity[g]
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        sum over items i of (unit_benefit_cents[i] * x[i] - item_fee_cents[i] * y[i])
      plus sum over bundles b of (bonus_cents[b] * w[b])
      minus sum over categories g of (activation_fee_cents[g] * z[g])
      (All terms in cents; bonuses and fees applied once per relevant activation.)
7.  **Formulate Constraints:**
    -   Constraint 1 (Item Order Bounds): For each authorized item i, x[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]]; for unauthorized items, x[i]=0.
    -   Constraint 2 (Item Activation): For each item i, y[i]=1 if x[i] ≥ minimum_lot[i], y[i]=0 if x[i]=0; enforce via x[i] ≥ minimum_lot[i]*y[i] and x[i] ≤ maximum_order[i]*y[i].
    -   Constraint 3 (Category Activation): For each category g, z[g]=1 if any item in g is ordered (i.e., sum over i in g of y[i] ≥ 1), z[g]=0 otherwise.
    -   Constraint 4 (Category Quantity Bounds): For each category g, sum over i in g of x[i] ≥ minimum_quantity[g] and ≤ maximum_quantity[g].
    -   Constraint 5 (Resource Capacity): For each resource r, sum over items i of (usage amount[i, r] * x[i] converted to base units) ≤ total available capacity for r (sum of opening and reservation entries, converted to base units).
    -   Constraint 6 (Incompatibility): For each incompatible pair (i, j), y[i] + y[j] ≤ 1.
    -   Constraint 7 (Prerequisite): For each prerequisite pair (i, j), y[i] ≤ y[j] (i requires j).
    -   Constraint 8 (Bundle Bonus): For each bundle b = (i, j), w[b] ≤ y[i], w[b] ≤ y[j], w[b] ≥ y[i] + y[j] - 1; w[b]=1 only if both items are ordered with positive quantity and both are authorized.
[Abstract Model Plan END]