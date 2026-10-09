[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (i.e., integer lot selection of authorized options) that yields the largest net benefit in USD cents, subject to a complex set of business rules: multi-currency benefit conversion, item and category fees, resource and category limits, option compatibility and dependency, and bundle bonuses. The data must be filtered for validity as of the date, using the highest revision per (dealership_id, table, record_id), excluding future events and deletes.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, and logical (compatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options/items (`i`): All valid item_ref for OSLO_NEW_CARS as of 2026-05-07.
    - Categories (`g`): All valid categories for OSLO_NEW_CARS as of 2026-05-07.
    - Resources (`r`): All valid resources (e.g., space, power, labor) for OSLO_NEW_CARS as of 2026-05-07.
    - Bundles (`b`): All valid bundle pairs for OSLO_NEW_CARS as of 2026-05-07.
    - Incompatible pairs (`(i,j)`): All valid incompatible option pairs.
    - Requires pairs (`(i,k)`): All valid requires (dependency) pairs.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of option/item `i` to order (must be 0 if unauthorized, otherwise between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `z[i]` = 1 if option/item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = 1 if any option in category `g` is selected (i.e., sum over i in g of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both options in bundle `b` are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit (in USD cents): For each item, sum all benefit components (from 'benefit' table) after converting each (amount * usd_cents_numerator / denominator) using the matching currency in the 'fx' table.
    -   Item fee (in USD cents): From 'item_fee' table, activation_fee_cents per item_ref.
    -   Category limits and activation fees: From 'category' table, minimum_quantity, maximum_quantity, activation_fee_cents per category.
    -   Resource usage per unit: From 'usage' table, amount and unit per (item_ref, resource); convert units to base (ml, minute, wh).
    -   Resource capacity: From 'capacity_ledger' table, sum of signed amounts per resource (convert units to base).
    -   Option authorization, lot sizes: From 'item' table, authorized (must be >0 to allow q[i]>0), minimum_lot, maximum_order, and category per item_ref.
    -   Incompatible pairs: From 'incompatible' table, all valid (item_a, item_b) pairs.
    -   Requires pairs: From 'requires' table, all valid (item_ref, prerequisite_ref) pairs.
    -   Bundle bonuses: From 'bundle' table, (item_a, item_b, bonus_cents).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all items: (benefit per unit in USD cents) * q[i]
    -   Minus: sum of item_fee for each option with q[i]>0 (i.e., once per selected option)
    -   Minus: sum of category activation_fee_cents for each category used (i.e., once per used category)
    -   Plus: sum of bundle bonus_cents for each bundle where both options are selected (once per bundle)
7.  **Formulate Constraints:**
    -   **Item selection and lot size:** For each item i:
        - If authorized[i] == 0, then q[i] = 0.
        - If authorized[i] > 0, then q[i] = 0 or minimum_lot[i] ≤ q[i] ≤ maximum_order[i].
        - z[i] = 1 if q[i] > 0, else 0.
    -   **Category quantity limits:** For each category g:
        - sum over i in g of q[i] ≥ minimum_quantity[g] * y[g]
        - sum over i in g of q[i] ≤ maximum_quantity[g] * y[g]
        - y[g] = 1 if any q[i]>0 for i in g, else 0.
    -   **Resource capacity:** For each resource r:
        - sum over i of (resource_usage[i,r] * q[i]) ≤ total_capacity[r]
        - All units converted to base (ml, minute, wh).
    -   **Incompatible options:** For each incompatible pair (i,j):
        - z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires dependencies:** For each requires pair (i,k):
        - q[i] ≤ M * z[k], where M is a large constant (or more strictly, q[k] ≥ 1 if q[i] ≥ 1).
    -   **Bundle bonuses:** For each bundle (i_a, i_b):
        - w[b] ≤ z[i_a], w[b] ≤ z[i_b], w[b] ≥ z[i_a] + z[i_b] - 1 (w[b]=1 iff both selected).
    -   **Item and category fees:** Deduct item_fee once per option with q[i]>0; deduct category activation_fee once per used category.
    -   **Integrality:** All q[i] are integer, z[i], y[g], w[b] are binary.
[Abstract Model Plan END]