[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer order quantities for each authorized bread option to maximize net benefit (in USD cents), considering per-unit benefits, fixed item and category activation fees, bundle bonuses, resource capacities (labor, space, power), category quantity bounds, incompatibilities, and prerequisite requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource allocation, and logical (combinatorial) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (i ∈ Items): All rows from the 'item' table with authorized=1.
    - Categories (c ∈ Categories): All rows from the 'category' table.
    - Resources (r ∈ Resources): All resources in the 'usage' and 'capacity_ledger' tables.
    - Incompatible pairs (p ∈ Incompatibles): All rows from the 'incompatible' table.
    - Prerequisite pairs (q ∈ Requires): All rows from the 'requires' table.
    - Bundles (b ∈ Bundles): All rows from the 'bundle' table.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item i (0, or between minimum_lot and maximum_order if positive). Type: GRB.INTEGER.
    -   `y[i]` = 1 if item i is ordered (x[i] ≥ minimum_lot), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is ordered (i.e., ∃i in c with y[i]=1), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle b are ordered (i.e., y[item_a]=1 and y[item_b]=1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - unit_benefit_cents (from 'item' table, per item i)
        - item_fee_cents (from 'item' table, per item i)
        - activation_fee_cents (from 'category' table, per category c)
        - bonus_cents (from 'bundle' table, per bundle b)
    -   Constraint coefficients:
        - minimum_lot, maximum_order (from 'item' table, per item i)
        - minimum_quantity, maximum_quantity (from 'category' table, per category c)
        - resource usage per unit (from 'usage' table: amount, unit, per item i and resource r)
        - resource capacity (from 'capacity_ledger' table: sum of amount per resource r, converted to consistent units)
    -   Logical constraints:
        - Incompatible pairs (from 'incompatible' table: item_a, item_b)
        - Prerequisites (from 'requires' table: item_ref, prerequisite_ref)
        - Category membership (from 'item' table: category)
        - Authorization (from 'item' table: authorized=1 filter)
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - Sum over all items: (unit_benefit_cents[i] * x[i]) − (item_fee_cents[i] * y[i])
    - Minus sum over all categories: (activation_fee_cents[c] * z[c])
    - Plus sum over all bundles: (bonus_cents[b] * w[b])
7.  **Formulate Constraints:**
    -   **Item selection and bounds:**
        - For each item i: x[i] = 0 or minimum_lot[i] ≤ x[i] ≤ maximum_order[i] (enforced via y[i])
        - For each item i: x[i] ≥ minimum_lot[i] * y[i]; x[i] ≤ maximum_order[i] * y[i]; x[i] ≥ 0; x[i] integer
        - For each item i: y[i] = 0 if authorized[i] ≠ 1
    -   **Category quantity bounds:**
        - For each category c: sum over i in c of x[i] ≥ minimum_quantity[c]
        - For each category c: sum over i in c of x[i] ≤ maximum_quantity[c]
        - For each category c: z[c] ≥ y[i] for all i in c; z[c] ∈ {0,1}
    -   **Resource capacity constraints:**
        - For each resource r: sum over i of (resource_usage[i,r] * x[i]) ≤ total_capacity[r] (with all units converted: 1000 ml = 1 liter, 60 min = 1 hour, 1000 wh = 1 kwh)
    -   **Incompatibility constraints:**
        - For each incompatible pair (item_a, item_b): y[item_a] + y[item_b] ≤ 1
    -   **Prerequisite constraints:**
        - For each requires pair (item, prerequisite): y[item] ≤ y[prerequisite]
    -   **Bundle bonus constraints:**
        - For each bundle b (item_a, item_b): w[b] ≤ y[item_a]; w[b] ≤ y[item_b]; w[b] ≥ y[item_a] + y[item_b] − 1; w[b] ∈ {0,1}
[Abstract Model Plan END]