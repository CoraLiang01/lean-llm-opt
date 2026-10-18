[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the integer replenishment quantities for each authorized option (item) for business unit NORTH, as of 2026-03-12, that maximize net benefit (in USD cents). The model must account for per-unit benefit, item and category activation fees, bundle bonuses, resource and category quantity limits, minimum/maximum order sizes, authorization, incompatibilities, and requires dependencies. All data tables must be filtered to the latest revision on or before the as-of date, excluding future events and deletions, with retransmissions deduplicated. All units must be converted as specified for resource comparisons.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (dependency/incompatibility) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Items (options) `i` (from item tables, after filtering)
    - Categories `g` (from category tables, after filtering)
    - Resources `r` (from resource/usage/capacity tables)
    - Bundles `b` (from bundle tables, each defined by a pair of items)
    - Incompatible pairs `(i,j)` (from incompatible tables)
    - Requires pairs `(i,k)` (from requires tables)
4.  **Define Decision Variables:**
    -   `q[i]` = Integer quantity ordered of item `i`. Type: GRB.INTEGER.
    -   `z[i]` = Binary, 1 if item `i` is selected (i.e., `q[i] > 0`), 0 otherwise. Type: GRB.BINARY.
    -   `y[g]` = Binary, 1 if any item in category `g` is selected, 0 otherwise. Type: GRB.BINARY.
    -   `w[b]` = Binary, 1 if both items in bundle `b` are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for item `i`: sum of all `amount_cents` for item `i` from all benefit tables, after filtering.
    -   Item activation fee: `activation_fee_cents` for item `i` from item_fee tables, after filtering.
    -   Category activation fee: `activation_fee_cents` for category `g` from category tables, after filtering.
    -   Bundle bonus: `bonus_cents` for bundle `b` from bundle tables, after filtering.
    -   Minimum/maximum order for item `i`: `minimum_lot`, `maximum_order` from item tables, after filtering.
    -   Authorization for item `i`: `authorized` from item tables, after filtering (must be positive to allow selection).
    -   Category quantity limits: `minimum_quantity`, `maximum_quantity` for category `g` from category tables, after filtering (apply unconditionally).
    -   Resource usage per unit for item `i` and resource `r`: `amount` from usage tables, after filtering and unit conversion.
    -   Resource capacity for resource `r`: sum of all signed `amount` for resource `r` from capacity_ledger tables, after filtering and unit conversion.
    -   Incompatible pairs: all pairs `(i,j)` from incompatible tables, after filtering.
    -   Requires pairs: all pairs `(i,k)` from requires tables, after filtering.
6.  **Formulate Objective:** Maximize total net benefit in USD cents:
        - Sum over items: (per-unit benefit) × `q[i]`
        - Minus sum over selected items: item activation fee × `z[i]` (charged once per item with positive quantity)
        - Minus sum over used categories: category activation fee × `y[g]` (charged once per category with any selected item)
        - Plus sum over bundles: bundle bonus × `w[b]` (awarded once per bundle if both items are selected)
7.  **Formulate Constraints:**
    -   **Item selection and order size:**
        - For each item `i`: `q[i] = 0` if not authorized; else, `minimum_lot[i] × z[i] ≤ q[i] ≤ maximum_order[i] × z[i]`, with `z[i] ∈ {0,1}`.
    -   **Category activation:** For each item `i` in category `g`: `z[i] ≤ y[g]`; for each category `g`: `y[g] ≤ sum(z[i] for i in g)`.
    -   **Category quantity limits:** For each category `g`: `minimum_quantity[g] ≤ sum(q[i] for i in g) ≤ maximum_quantity[g]` (apply even if no items are selected).
    -   **Resource capacity:** For each resource `r`: sum over items of (resource usage per unit for `i` and `r` × `q[i]`) ≤ total capacity for `r` (after unit conversion).
    -   **Incompatibility:** For each incompatible pair `(i,j)`: `z[i] + z[j] ≤ 1`.
    -   **Requires dependencies:** For each requires pair `(i,k)`: `q[i] > 0 ⇒ q[k] > 0` (enforce with `z[i] ≤ z[k]`).
    -   **Bundle bonus activation:** For each bundle `b` with items `(i,j)`: `w[b] ≤ z[i]`, `w[b] ≤ z[j]`, `w[b] ≥ z[i] + z[j] - 1`.
    -   **Variable domains:** All `q[i]` are integer ≥ 0; all `z[i]`, `y[g]`, `w[b]` are binary.
[Abstract Model Plan END]