[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, to maximize net benefit (in USD cents). The model must account for per-unit benefits, item and category activation fees, resource and category quantity limits, option compatibility and dependency rules, and bundle bonuses. Only the latest non-future, non-DELETE revision for each (tenant, table, record_id) is used; unauthorized options must have zero quantity.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (compatibility/dependency) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables for tenant NORTH, filtered as per revision rules)
    - Categories `g` (from category tables for tenant NORTH)
    - Resources `r` (from usage/capacity tables for tenant NORTH)
    - Bundles `b` (from bundle tables for tenant NORTH)
    - Incompatible pairs and requires pairs (from respective tables for tenant NORTH)
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of item `i`. Type: GRB.INTEGER.
    -   `z[i]` = 1 if item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    -   `w[g]` = 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `s[b]` = 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit: sum of `amount_cents` from benefit tables (by item_ref, after selection rules).
    -   Item activation fee: `activation_fee_cents` from item_fee tables (by item_ref, after selection rules).
    -   Category activation fee: `activation_fee_cents` from category tables (by category, after selection rules).
    -   Bundle bonus: `bonus_cents` from bundle tables (by item_a, item_b, after selection rules).
    -   Resource usage per unit: `amount` and `unit` from usage tables (by item_ref, resource, after selection rules).
    -   Resource capacity: sum of `amount` (converted to base units) from capacity_ledger tables (by resource, after selection rules).
    -   Item authorization, minimum_lot, maximum_order, category: from item tables (by item_ref, after selection rules).
    -   Category min/max quantity and activation fee: from category tables (by category, after selection rules).
    -   Incompatible pairs: from incompatible tables (by item_a, item_b, after selection rules).
    -   Requires pairs: from requires tables (by item_ref, prerequisite_ref, after selection rules).
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
    -   Sum over items: (per-unit benefit) × `q[i]`
    -   Minus: sum of item activation fees for each item with `q[i] > 0`
    -   Minus: sum of category activation fees for each category with any item selected
    -   Plus: sum of bundle bonuses for each bundle where both items are selected
7.  **Formulate Constraints:**
    -   **Data Selection:** For each table, use only rows for tenant NORTH, with effective_date ≤ 2026-03-12, highest revision per (table, record_id), and exclude if action is DELETE. Apply this before any joins or aggregations.
    -   **Authorization:** For each item, if authorized = 0, enforce `q[i] = 0`.
    -   **Item Quantity Bounds:** For each item, enforce `minimum_lot[i] ≤ q[i] ≤ maximum_order[i]` if authorized; else `q[i] = 0`. All `q[i]` are integer.
    -   **Category Quantity Bounds:** For each category, sum of `q[i]` over items in category `g` must satisfy `minimum_quantity[g] ≤ sum_i_in_g q[i] ≤ maximum_quantity[g]` (unconditional).
    -   **Resource Capacity:** For each resource, sum over items of (resource usage per unit × `q[i]`, converted to base units) ≤ total available capacity (converted to base units).
    -   **Item Activation:** For each item, `z[i] = 1` if `q[i] > 0`, else `z[i] = 0`.
    -   **Category Activation:** For each category, `w[g] = 1` if any `q[i] > 0` for items in `g`, else `w[g] = 0`.
    -   **Incompatibility:** For each incompatible pair (i, j), enforce `z[i] + z[j] ≤ 1`.
    -   **Requires Dependency:** For each requires pair (i, j), enforce `q[i] > 0` ⇒ `q[j] > 0` (i.e., `z[i] ≤ z[j]`).
    -   **Bundle Bonus:** For each bundle (i, j), `s[b] = 1` if both `z[i] = 1` and `z[j] = 1`, else `s[b] = 0`.
    -   **Integrality:** All `q[i]` are integer, all `z[i]`, `w[g]`, `s[b]` are binary.
[Abstract Model Plan END]