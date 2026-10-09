[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (i.e., integer lot selection of authorized options) that yields the largest net benefit in USD cents, subject to a complex set of business rules. This involves: (a) filtering all relevant tables to valid, non-deleted, latest-revision records as of the cutoff date; (b) converting all benefit components to USD cents using the appropriate FX rates; (c) deducting item and category activation fees as specified; (d) enforcing resource, category, incompatibility, and dependency constraints; and (e) awarding bundle bonuses as appropriate.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options/Items (`i`): Each vehicle option/item that can be ordered.
    - Categories (`g`): Each category grouping options/items.
    - Resources (`r`): Each resource type (e.g., space, power, labor).
    - Bundles (`b`): Each bundle of two options eligible for a bonus.
    - Incompatible pairs (`(i,j)`): Pairs of options that cannot be ordered together.
    - Requires pairs (`(i,k)`): Pairs where option `i` requires option `k`.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of option/item `i` to order (must be 0 if unauthorized; otherwise between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `z[i]` = 1 if option/item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = 1 if any option in category `g` is selected (i.e., sum over i in g of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both options in bundle `b` are selected (i.e., both z[i_a] and z[i_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit components per item (in various currencies): from 'benefit' tables, columns 'amount', 'currency', 'item_ref', 'component'.
    -   FX rates: from 'fx' table, columns 'currency', 'usd_cents_numerator', 'denominator'.
    -   Item fees: from 'item_fee' table, columns 'item_ref', 'activation_fee_cents'.
    -   Category limits and activation fees: from 'category' table, columns 'category', 'minimum_quantity', 'maximum_quantity', 'activation_fee_cents'.
    -   Item authorization, lot sizes, and category: from 'item' table, columns 'item_ref', 'authorized', 'minimum_lot', 'maximum_order', 'category'.
    -   Resource usage per unit: from 'usage' table, columns 'item_ref', 'resource', 'amount', 'unit'.
    -   Resource capacity: from 'capacity_ledger' table, columns 'resource', 'amount', 'unit', summed over all valid records.
    -   Incompatible pairs: from 'incompatible' table, columns 'item_a', 'item_b'.
    -   Requires pairs: from 'requires' table, columns 'item_ref', 'prerequisite_ref'.
    -   Bundle bonuses: from 'bundle' table, columns 'item_a', 'item_b', 'bonus_cents'.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   Sum over all items: (sum of all benefit components for item i, converted to USD cents) × q[i]
    -   Minus: sum over all selected options with q[i] > 0 of item_fee[i]
    -   Minus: sum over all used categories of category activation_fee[g]
    -   Plus: sum over all awarded bundle bonuses (for each bundle where both options are selected)
7.  **Formulate Constraints:**
    -   **Data Validity:** For each table, only include records for OSLO_NEW_CARS, with effective_date ≤ 2026-05-07, highest integer revision per (dealership_id, table, record_id), and action ≠ DELETE.
    -   **Authorization:** For each option/item i, q[i] = 0 if authorized[i] == 0; otherwise, minimum_lot[i] ≤ q[i] ≤ maximum_order[i] or q[i] = 0.
    -   **Item-Selection Linking:** z[i] = 1 if q[i] > 0, 0 otherwise.
    -   **Category Quantity Limits:** For each category g, sum over i in g of q[i] between minimum_quantity[g] and maximum_quantity[g] if any q[i] > 0; otherwise, zero.
    -   **Category Activation:** y[g] = 1 if any q[i] > 0 for i in g, 0 otherwise.
    -   **Resource Constraints:** For each resource r, sum over i of (resource_usage[i,r] × q[i]) ≤ total_capacity[r], after converting all units to a common base (e.g., liters to ml, hours to minutes, kwh to wh).
    -   **Incompatibility:** For each incompatible pair (i,j), z[i] + z[j] ≤ 1.
    -   **Requires Dependencies:** For each requires pair (i,k), q[i] > 0 ⇒ q[k] > 0 (i.e., z[i] ≤ z[k]).
    -   **Bundle Bonuses:** For each bundle (i_a, i_b), w[b] = 1 if z[i_a] = 1 and z[i_b] = 1; w[b] = 0 otherwise.
    -   **Item Fee Deduction:** Deduct item_fee[i] once for each i with q[i] > 0.
    -   **Category Fee Deduction:** Deduct category activation_fee[g] once for each g with any q[i] > 0.
    -   **Bundle Bonus Award:** Award bundle bonus only if both options are selected and both are authorized.
    -   **Integrality:** All q[i] are integer, z[i], y[g], w[b] are binary.
[Abstract Model Plan END]