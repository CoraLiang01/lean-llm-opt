[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit in USD cents. The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option incompatibilities and dependencies, bundle bonuses, and unit conversions for resource usage and capacity. Only the latest (highest revision) non-future, non-DELETE record for each (tenant, table, record_id) is used, and identical retransmissions are counted once.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`i`): All item_ref values for authorized options in NORTH, as of 2026-03-12.
    - Categories (`g`): All categories associated with items in NORTH.
    - Resources (`r`): All resources (e.g., labor, space, power) used by items in NORTH.
    - Bundles (`b`): All bundle pairs (item_a, item_b) in NORTH.
    - Incompatibility pairs (`(i,j)`): All incompatible item pairs in NORTH.
    - Requires pairs (`(i,k)`): All requires (dependency) pairs in NORTH.
4.  **Define Decision Variables:**
    -   `x[i]` = Integer quantity ordered of item/option `i`. Type: GRB.INTEGER.
    -   `y[i]` = 1 if item/option `i` is ordered (i.e., x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[g]` = 1 if any item in category `g` is ordered (i.e., sum_{i in g} x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both items in bundle `b` are ordered (i.e., x[item_a] > 0 and x[item_b] > 0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: sum of amount_cents from all benefit rows for each item_ref (from benefit tables, filtered as per selection rules).
    -   Item activation fee: activation_fee_cents from item_fee table for each item_ref.
    -   Category activation fee: activation_fee_cents from category table for each category.
    -   Minimum/maximum order per item: minimum_lot, maximum_order from item tables.
    -   Authorization: authorized field from item tables (must be >0 for option to be eligible).
    -   Resource usage per unit: amount and unit from usage tables for each item_ref and resource.
    -   Resource capacity: sum of amount (with sign) from capacity_ledger for each resource, converted to base units (ml, wh, minute).
    -   Category quantity limits: minimum_quantity, maximum_quantity from category tables.
    -   Incompatibility: item_a, item_b pairs from incompatible tables.
    -   Requires dependencies: item_ref, prerequisite_ref pairs from requires tables.
    -   Bundle bonuses: bonus_cents from bundle tables for each (item_a, item_b) pair.
    -   Unit conversions: 1000 ml = 1 liter, 60 minutes = 1 hour, 1000 wh = 1 kwh.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    -   sum over items: (per-unit benefit[i] * x[i])
    -   minus sum over items: (item activation_fee[i] * y[i]) for each item with x[i] > 0
    -   minus sum over categories: (category activation_fee[g] * z[g]) for each category used
    -   plus sum over bundles: (bundle_bonus[b] * w[b]) for each bundle where both items are ordered
7.  **Formulate Constraints:**
    -   **Data Selection:** For each table, only use rows for tenant == 'NORTH' and effective_date <= '2026-03-12'. For each (tenant, table, record_id), keep only the row with the highest revision (integer), and discard if action == 'DELETE'. Identical retransmissions count once.
    -   **Authorization:** For each item, if authorized == 0, enforce x[i] = 0.
    -   **Item Order Bounds:** For each item, minimum_lot[i] * y[i] <= x[i] <= maximum_order[i] * y[i]; x[i] = 0 if y[i] = 0.
    -   **Category Quantity Limits:** For each category g, sum_{i in g} x[i] >= minimum_quantity[g] * z[g] and sum_{i in g} x[i] <= maximum_quantity[g] * z[g].
    -   **Resource Capacity:** For each resource r, sum_{i} (resource_usage[i,r] * x[i]) <= resource_capacity[r], with all units converted to base units (ml, wh, minute).
    -   **Category Activation:** z[g] = 1 if any x[i] > 0 for i in category g; z[g] = 0 otherwise.
    -   **Incompatibility:** For each incompatible pair (i,j), y[i] + y[j] <= 1 (cannot order both).
    -   **Requires Dependencies:** For each requires pair (i,k), y[i] <= y[k] and x[i] > 0 => x[k] > 0 (prerequisite must be ordered in positive quantity if dependent is ordered).
    -   **Bundle Bonuses:** For each bundle (item_a, item_b), w[b] = 1 if y[item_a] = 1 and y[item_b] = 1; w[b] = 0 otherwise. Award bonus only if both are ordered in positive quantity and both are authorized.
    -   **Variable Domains:** x[i] >= 0 and integer; y[i], z[g], w[b] in {0,1}.
[Abstract Model Plan END]