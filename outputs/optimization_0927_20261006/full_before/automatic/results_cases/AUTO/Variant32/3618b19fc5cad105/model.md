[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (i.e., integer lot selection of authorized options) that yields the largest net benefit in USD cents. This must account for multi-currency benefit components, item and category activation fees, bundle bonuses, resource and category quantity limits, option incompatibilities, and requires dependencies. The model must use only the latest non-future, non-deleted records per (dealership_id, table, record_id), and all monetary values must be converted to USD cents using the provided FX table.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, logical constraints, and resource/capacity limits.
3.  **Define Index Sets:** The primary indices are:
    - Options/Items (`i`): All authorized options/items for OSLO_NEW_CARS as of 2026-05-07.
    - Categories (`g`): All categories for OSLO_NEW_CARS as of 2026-05-07.
    - Resources (`r`): All resources (e.g., space, power, labor) for OSLO_NEW_CARS as of 2026-05-07.
    - Bundles (`b`): All valid bundle pairs for OSLO_NEW_CARS as of 2026-05-07.
    - Incompatible pairs (`(i,j)`): All incompatible option pairs.
    - Requires pairs (`(i,k)`): All requires dependencies (option i requires option k).
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of option/item `i` to order. Type: GRB.INTEGER. Must be 0 if unauthorized; otherwise, between minimum_lot and maximum_order.
    -   `z[i]` = Binary variable: 1 if option/item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = Binary variable: 1 if any option in category `g` is selected (i.e., sum over i in g of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary variable: 1 if both options in bundle `b` are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit components: From `benefit` table, columns `amount`, `currency`, `item_ref`, `component`. Convert each to USD cents using the latest FX rate from the `fx` table (`usd_cents_numerator`/`denominator`).
    -   Item fees: From `item_fee` table, columns `item_ref`, `activation_fee_cents`.
    -   Category limits and activation fees: From `category` table, columns `category`, `minimum_quantity`, `maximum_quantity`, `activation_fee_cents`.
    -   Option authorization, lot sizes: From `item` table, columns `item_ref`, `authorized`, `minimum_lot`, `maximum_order`, `category`.
    -   Resource usage per unit: From `usage` table, columns `item_ref`, `resource`, `amount`, `unit` (convert units as needed: 1000 ml = 1 liter, 60 min = 1 hour, 1000 wh = 1 kwh).
    -   Resource capacity: From `capacity_ledger` table, sum of `amount` for each resource (opening + reservation), after unit conversion.
    -   Incompatible pairs: From `incompatible` table, columns `item_a`, `item_b`.
    -   Requires dependencies: From `requires` table, columns `item_ref`, `prerequisite_ref`.
    -   Bundle bonuses: From `bundle` table, columns `item_a`, `item_b`, `bonus_cents`.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    - For each option, sum all benefit components (converted to USD cents) per unit, multiply by q[i].
    - Subtract item_fee for each option with q[i] > 0 (i.e., for each z[i]=1).
    - Subtract category activation_fee for each category used (i.e., for each y[g]=1).
    - Add bundle bonus for each bundle where both options are selected (i.e., for each w[b]=1).
    - Objective: Maximize [sum over i of (total_benefit_per_unit[i] * q[i] - item_fee[i] * z[i]) + sum over b of bundle_bonus[b] * w[b] - sum over g of category_fee[g] * y[g]]
7.  **Formulate Constraints:**
    -   **Authorization and Lot Constraints:** For each option i:
        - If authorized[i] == 0, q[i] = 0.
        - If authorized[i] > 0, minimum_lot[i] * z[i] ≤ q[i] ≤ maximum_order[i] * z[i]; z[i] ∈ {0,1}.
    -   **Category Quantity Limits:** For each category g:
        - minimum_quantity[g] * y[g] ≤ sum over i in g of q[i] ≤ maximum_quantity[g] * y[g]; y[g] ∈ {0,1}.
        - For each i in g: z[i] ≤ y[g] (if any option in g is selected, y[g]=1).
    -   **Resource Capacity Constraints:** For each resource r:
        - sum over i of (resource_usage_per_unit[i,r] * q[i]) ≤ total_capacity[r] (after all unit conversions).
    -   **Incompatibility Constraints:** For each incompatible pair (i,j):
        - z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires Constraints:** For each requires pair (i,k):
        - q[i] ≤ maximum_order[i] * (q[k] ≥ 1) (i.e., if q[i] > 0, then q[k] ≥ 1).
        - Implemented as: q[i] ≤ maximum_order[i] * y_req[k], where y_req[k]=1 if q[k] ≥ 1, else 0.
    -   **Bundle Bonus Constraints:** For each bundle (i_a, i_b):
        - w[b] ≤ z[i_a], w[b] ≤ z[i_b], w[b] ≥ z[i_a] + z[i_b] - 1 (w[b]=1 iff both options are selected).
    -   **Item Fee Application:** For each option i, item_fee is charged only if z[i]=1 (i.e., q[i] > 0).
    -   **Category Fee Application:** For each category g, activation_fee is charged only if y[g]=1 (i.e., any option in g is selected).
    -   **Variable Domains:** q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer), z[i], y[g], w[b] ∈ {0,1}.
[Abstract Model Plan END]