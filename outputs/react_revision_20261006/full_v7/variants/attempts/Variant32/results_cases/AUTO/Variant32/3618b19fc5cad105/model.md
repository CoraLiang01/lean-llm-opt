[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (i.e., integer lot selection of authorized options) that yields the largest net benefit in USD cents, considering all benefit components (converted to USD), item and category activation fees, bundle bonuses, resource and category quantity limits, option compatibility and dependency constraints, and resource usage/capacity (with unit conversions).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, compatibility, and dependency (logic) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options/items (`i`): All valid, authorized options for OSLO_NEW_CARS as of 2026-05-07.
    - Categories (`g`): All valid categories for OSLO_NEW_CARS as of 2026-05-07.
    - Resources (`r`): All resources (e.g., space, power, labor) relevant to OSLO_NEW_CARS.
    - Bundles (`b`): All valid bundle bonus pairs for OSLO_NEW_CARS.
    - Incompatible pairs (`(i,j)`): All valid incompatible option pairs.
    - Requires pairs (`(i,k)`): All valid requires (dependency) pairs.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of option/item `i` to order. Type: GRB.INTEGER. Must be 0 if unauthorized; otherwise, between minimum_lot and maximum_order.
    -   `z[i]` = 1 if option/item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = 1 if any option in category `g` is selected (i.e., sum over i in g of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both options in bundle `b` are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each option: sum of all benefit components (from 'benefit' table), converted to USD cents using the latest valid FX rate as of 2026-05-07 (from 'fx' table: amount * usd_cents_numerator / denominator).
    -   Item activation fee: from 'item_fee' table, per option.
    -   Category activation fee: from 'category' table, per category.
    -   Bundle bonus: from 'bundle' table, per valid bundle.
    -   Resource usage per unit: from 'usage' table, per option and resource, with unit conversion (liter→ml, hour→minute, kwh→wh).
    -   Resource capacity: sum of 'capacity_ledger' entries (opening/reservation) for each resource, as of 2026-05-07, with unit conversion.
    -   Option authorization, minimum_lot, maximum_order, and category assignment: from 'item' table.
    -   Category quantity limits and activation fees: from 'category' table.
    -   Incompatible pairs: from 'incompatible' table.
    -   Requires pairs: from 'requires' table.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all options: (benefit per unit in USD cents) × q[i]
    -   Minus: sum of item activation fees for each option with q[i] > 0 (deducted once per option)
    -   Minus: sum of category activation fees for each category with any option selected (deducted once per category)
    -   Plus: sum of bundle bonuses for each bundle where both options are selected (awarded once per bundle)
7.  **Formulate Constraints:**
    -   **Option Quantity Bounds:** For each option `i`:
        - If authorized: minimum_lot[i] ≤ q[i] ≤ maximum_order[i]; if not authorized: q[i] = 0.
    -   **Item Activation Indicator:** For each option `i`: z[i] = 1 if q[i] ≥ 1, 0 otherwise (enforced via q[i] ≥ z[i] × minimum_lot[i], q[i] ≤ z[i] × maximum_order[i]).
    -   **Category Activation Indicator:** For each category `g`: y[g] = 1 if any z[i] = 1 for i in g; y[g] ≥ z[i] for all i in g.
    -   **Category Quantity Limits:** For each category `g`: sum over i in g of q[i] ≤ maximum_quantity[g] and ≥ minimum_quantity[g] (from 'category' table).
    -   **Resource Capacity Constraints:** For each resource `r`: sum over i of (resource usage per unit for i and r × q[i]) ≤ total available capacity for r (after unit conversion).
    -   **Incompatible Pairs:** For each incompatible pair (i, j): z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires Pairs:** For each requires pair (i, k): q[i] ≤ maximum_order[i] × z[k] (i can only be selected if k is selected with positive quantity).
    -   **Bundle Bonus Indicator:** For each bundle (i_a, i_b): w[b] ≤ z[i_a], w[b] ≤ z[i_b], w[b] ≥ z[i_a] + z[i_b] - 1 (w[b] = 1 iff both options are selected).
    -   **Integrality:** All q[i] are integer, all z[i], y[g], w[b] are binary.
    -   **Other unconditional bounds:** All variables are nonnegative; all constraints apply unconditionally unless the query specifies otherwise.
[Abstract Model Plan END]