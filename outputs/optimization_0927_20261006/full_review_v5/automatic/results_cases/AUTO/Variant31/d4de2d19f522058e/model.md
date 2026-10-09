[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select integer quantities of authorized vehicle configurations for delivery, maximizing total net benefit (unit benefit minus fixed item and category fees, plus bundle bonuses), subject to resource, category, compatibility, and dependency constraints, using only the supplied configuration rows.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (compatibility, dependency, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (vehicle configurations) from the 'item' table (item_ref).
    - Categories from the 'category' table (category).
    - Resources from the 'capacity_ledger' and 'usage' tables (resource).
    - Bundles from the 'bundle' table (pairs of items).
    - Incompatible pairs from the 'incompatible' table (pairs of items).
    - Requires dependencies from the 'requires' table (item, prerequisite).
4.  **Define Decision Variables:**
    - `q[i]` = Integer quantity of item i selected for delivery (i in Items). Type: GRB.INTEGER.
    - `z[i]` = 1 if item i is selected (q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    - `y[c]` = 1 if any item in category c is selected, 0 otherwise. Type: GRB.BINARY.
    - `b[bundle]` = 1 if both items in bundle are selected (for each bundle), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients:
        - 'unit_benefit_cents' (per unit, from 'item' table).
        - 'item_fee_cents' (fixed per item if selected, from 'item' table).
        - 'activation_fee_cents' (fixed per category if any item selected, from 'category' table).
        - 'bonus_cents' (per bundle, from 'bundle' table).
    - Constraint coefficients:
        - 'amount' (per-unit resource usage, from 'usage' table, by item and resource).
        - 'amount' (resource capacity, from 'capacity_ledger' table, sum of 'opening' and 'reservation' per resource).
        - 'minimum_lot', 'maximum_order' (per item, from 'item' table).
        - 'authorized' (per item, from 'item' table).
        - 'minimum_quantity', 'maximum_quantity' (per category, from 'category' table).
    - Logical relationships:
        - Incompatible pairs (from 'incompatible' table).
        - Requires dependencies (from 'requires' table).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        sum over items of (unit_benefit_cents[i] * q[i] - item_fee_cents[i] * z[i])
      minus sum over categories of (activation_fee_cents[c] * y[c])
      plus sum over bundles of (bonus_cents[bundle] * b[bundle]).
7.  **Formulate Constraints:**
    - **Item selection and bounds:**
        - For each item i: If authorized[i] = 1, minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]; if authorized[i] = 0, q[i] = 0 and z[i] = 0.
        - z[i] = 1 if q[i] ≥ 1, 0 otherwise.
    - **Resource limits:**
        - For each resource r: sum over items of (usage_amount[i,r] * q[i]) ≤ total_capacity[r], where total_capacity[r] = sum of 'amount' for 'opening' and 'reservation' entries in 'capacity_ledger' for resource r.
    - **Category quantity limits and activation:**
        - For each category c: minimum_quantity[c] ≤ sum over items in c of q[i] ≤ maximum_quantity[c].
        - For each item i in category c: z[i] ≤ y[c]; for each category c: y[c] ≤ sum over items in c of z[i].
    - **Incompatible pairs:**
        - For each incompatible pair (i, j): z[i] + z[j] ≤ 1.
    - **Requires dependencies:**
        - For each (i, prerequisite): z[i] ≤ z[prerequisite].
    - **Bundle bonuses:**
        - For each bundle (i, j): b[bundle] ≤ z[i], b[bundle] ≤ z[j], b[bundle] ≥ z[i] + z[j] - 1.
[Abstract Model Plan END]