[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select, as of 2026-05-07, the valid OSLO_NEW_CARS vehicle order (configuration) with the largest net benefit, considering all business rules: multi-currency benefit conversion, item and category fees, resource and category limits, option authorization, minimum/maximum lot sizes, incompatibilities, requires dependencies, and bundle bonuses. The model must apply a precise data selection rule for each table (filtering by date, revision, and DELETEs), sum all benefit components per item in USD cents, and deduct/award all relevant fees and bonuses exactly once per their rules.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed charges, logical constraints, and resource/capacity limits.
3.  **Define Index Sets:** The primary indices are:
    - Items/options (`i`): All valid item_ref for OSLO_NEW_CARS as of 2026-05-07.
    - Categories (`g`): All valid categories for OSLO_NEW_CARS as of 2026-05-07.
    - Resources (`r`): All valid resources (e.g., space, power, labor) for OSLO_NEW_CARS as of 2026-05-07.
    - Bundles (`b`): All valid bundle pairs (item_a, item_b) for OSLO_NEW_CARS as of 2026-05-07.
    - Incompatibility pairs (`(i,j)`): All valid incompatible item pairs for OSLO_NEW_CARS as of 2026-05-07.
    - Requires pairs (`(i,k)`): All valid requires (item, prerequisite) pairs for OSLO_NEW_CARS as of 2026-05-07.
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity of item/option `i` to order. Type: GRB.INTEGER. Must be 0 if unauthorized.
    -   `z[i]` = Binary, 1 if item/option `i` is selected (i.e., q[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = Binary, 1 if any item in category `g` is selected (i.e., sum over i in g of z[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary, 1 if both items in bundle `b` are selected (i.e., z[item_a] = z[item_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Benefit per unit for each item (`benefit[i]`): Sum of all benefit components for item `i` (from benefit tables), converted to USD cents using the latest valid FX rate as of 2026-05-07 (from fx table: amount * usd_cents_numerator / denominator).
    -   Item fee for each item (`item_fee[i]`): From item_fee table, activation_fee_cents for item_ref `i`.
    -   Category limits and activation fees (`min_g`, `max_g`, `cat_fee[g]`): From category table, minimum_quantity, maximum_quantity, activation_fee_cents for category `g`.
    -   Item authorization, lot sizes, and order limits (`auth[i]`, `min_lot[i]`, `max_order[i]`, `cat[i]`): From item table, authorized, minimum_lot, maximum_order, and category for item `i`.
    -   Resource usage per unit (`usage[i,r]`): From usage table, amount for item_ref `i` and resource `r`, converted to base units (liter→ml, hour→minute, kwh→wh).
    -   Resource capacity (`cap[r]`): From capacity_ledger table, sum of all valid signed amounts for resource `r` (opening + reservation), in base units.
    -   Incompatibility pairs: From incompatible table, all valid (item_a, item_b) pairs.
    -   Requires pairs: From requires table, all valid (item_ref, prerequisite_ref) pairs.
    -   Bundle bonuses (`bonus[b]`): From bundle table, bonus_cents for each valid bundle (item_a, item_b).
6.  **Formulate Objective:** Maximize net benefit in USD cents:
    -   Total benefit = sum over items of (benefit[i] * q[i])
    -   Minus sum over selected items of item_fee[i] (charged once per item with q[i] > 0)
    -   Minus sum over used categories of cat_fee[g] (charged once per category with any item selected)
    -   Plus sum over awarded bundles of bonus[b] (awarded once per bundle if both items are selected)
    -   Objective: Maximize [sum_i (benefit[i] * q[i]) - sum_i (item_fee[i] * z[i]) - sum_g (cat_fee[g] * y[g]) + sum_b (bonus[b] * w[b])]
7.  **Formulate Constraints:**
    -   **Authorization:** For each item `i`, if not authorized (auth[i] == 0), then q[i] = 0.
    -   **Lot and Order Limits:** For each item `i`, if authorized, then q[i] = 0 or min_lot[i] ≤ q[i] ≤ max_order[i], and q[i] is integer.
    -   **Category Quantity Limits:** For each category `g`, sum over i in g of q[i] ≥ min_g and ≤ max_g (unconditional, i.e., applies even if no items are selected).
    -   **Category Activation:** For each category `g`, y[g] = 1 if any q[i] > 0 for i in g, else 0.
    -   **Item Activation:** For each item `i`, z[i] = 1 if q[i] > 0, else 0.
    -   **Resource Capacity:** For each resource `r`, sum over items of (usage[i,r] * q[i]) ≤ cap[r]. All units converted to base (1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    -   **Incompatibility:** For each incompatible pair (i, j), z[i] + z[j] ≤ 1 (cannot select both).
    -   **Requires:** For each requires pair (i, k), q[i] > 0 ⇒ q[k] > 0 (i.e., z[i] ≤ z[k]), but no ratio required.
    -   **Bundle Bonus:** For each bundle (item_a, item_b), w[b] = 1 if z[item_a] = z[item_b] = 1, else 0; bonus awarded only if both are selected and both are authorized.
    -   **Item/Category/Bundle Fees/Bonuses:** Each fee/bonus is charged/awarded at most once per item/category/bundle, regardless of quantity.
    -   **Integrality:** All q[i] are integer, all z[i], y[g], w[b] are binary.
[Abstract Model Plan END]