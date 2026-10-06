[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (i.e., integer lot selection of authorized options) that yields the largest net benefit in USD cents, subject to a complex set of business rules: multi-currency benefit conversion, item and category fees, resource and category limits, option compatibility and dependency, and bundle bonuses. The data must be filtered for validity as of the cutoff date, using the highest revision per (dealership_id, table, record_id), excluding future events and DELETEs, and deduplicating retransmissions.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, resource constraints, and logical (compatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options/items (`i`): All valid, authorized item options for OSLO_NEW_CARS as of 2026-05-07.
    - Categories (`g`): All valid categories for OSLO_NEW_CARS as of 2026-05-07.
    - Resources (`r`): All relevant resources (e.g., space, power, labor) for OSLO_NEW_CARS.
    - Bundles (`b`): All valid bundle bonus pairs for OSLO_NEW_CARS.
    - Incompatible pairs (`(i,j)`): All valid incompatible option pairs.
    - Requires pairs (`(i,k)`): All valid requires (dependency) pairs.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of option/item `i` to order (must be 0 if unauthorized; otherwise between minimum_lot and maximum_order). Type: GRB.INTEGER.
    -   `z[i]` = 1 if option/item `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = 1 if any option in category `g` is selected (i.e., sum over i in g of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = 1 if both options in bundle `b` are selected (i.e., both z[i_a] and z[i_b] are 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   **Benefit per unit (USD cents):** For each item option, sum all valid benefit components (from 'benefit' table), converting each amount to USD cents using the corresponding FX rate (from 'fx' table: amount * usd_cents_numerator / denominator), as of the cutoff date.
    -   **Item fee (USD cents):** For each option, from 'item_fee' table, activation_fee_cents (charged once per option if q[i] > 0).
    -   **Category limits and activation fee:** From 'category' table: minimum_quantity, maximum_quantity, activation_fee_cents (charged once per category if any option in the category is selected).
    -   **Resource usage per unit:** From 'usage' table: for each item and resource, amount and unit (convert to base units: 1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    -   **Resource capacity:** From 'capacity_ledger' table: sum all valid entries for each resource (opening + reservation, in base units).
    -   **Authorization, lot size, max order:** From 'item' table: authorized (must be >0 to allow q[i]>0), minimum_lot, maximum_order, category.
    -   **Incompatible pairs:** From 'incompatible' table: pairs of item_refs that cannot both be selected.
    -   **Requires pairs:** From 'requires' table: (item_ref, prerequisite_ref) pairs; if item_ref is selected, prerequisite_ref must have q>0.
    -   **Bundle bonuses:** From 'bundle' table: (item_a, item_b, bonus_cents), awarded once if both options are selected.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over all options: (benefit per unit in USD cents) * q[i]
    -   Minus: sum over all selected options: item_fee[i] * z[i]
    -   Minus: sum over all used categories: activation_fee_cents[g] * y[g]
    -   Plus: sum over all awarded bundles: bonus_cents[b] * w[b]
7.  **Formulate Constraints:**
    -   **Data Validity:** For each table, only include records for OSLO_NEW_CARS with effective_date ≤ 2026-05-07, highest revision per (dealership_id, table, record_id), and action ≠ DELETE. Deduplicate retransmissions.
    -   **Authorization and Lot Constraints:** For each option i:
        -   If authorized[i] == 0, q[i] = 0.
        -   If authorized[i] > 0, q[i] ∈ {0} ∪ [minimum_lot[i], maximum_order[i]] (integer).
        -   z[i] = 1 if q[i] > 0, else 0.
    -   **Category Quantity Limits:** For each category g:
        -   sum over i in g of q[i] ≥ minimum_quantity[g] * y[g]
        -   sum over i in g of q[i] ≤ maximum_quantity[g] * y[g]
        -   y[g] = 1 if any q[i] > 0 for i in g, else 0.
    -   **Resource Capacity Constraints:** For each resource r:
        -   sum over i of (resource_usage[i,r] * q[i]) ≤ total_capacity[r]
        -   All units converted to base units (ml, minute, wh).
    -   **Incompatibility Constraints:** For each incompatible pair (i,j):
        -   z[i] + z[j] ≤ 1 (cannot both be selected).
    -   **Requires Constraints:** For each requires pair (i,k):
        -   q[i] > 0 ⇒ q[k] > 0 (can be modeled as q[i] ≤ M * z[k], with M large enough).
    -   **Bundle Bonus Constraints:** For each bundle (i_a, i_b):
        -   w[b] ≤ z[i_a], w[b] ≤ z[i_b], w[b] ≥ z[i_a] + z[i_b] - 1 (w[b]=1 iff both selected).
    -   **Item Fee Application:** For each option i:
        -   Item fee is charged once if q[i] > 0 (modeled via z[i]).
    -   **Category Activation Fee:** For each category g:
        -   Activation fee is charged once if any q[i] > 0 for i in g (modeled via y[g]).
    -   **Variable Domains:** q[i] ∈ ℤ₊, z[i], y[g], w[b] ∈ {0,1}.
[Abstract Model Plan END]