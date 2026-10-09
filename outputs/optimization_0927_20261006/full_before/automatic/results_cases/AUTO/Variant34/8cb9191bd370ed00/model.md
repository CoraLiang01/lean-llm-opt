[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each product-and-section option (item_ref, location_id) in MARKET_SQUARE, maximizing net merchandising benefit. The plan must respect section capacities, per-option authorization and lot/order limits, resource usage, category quantity bounds, fixed activation fees (per option and per category), incompatibility and prerequisite (requires) relationships, and bundle bonuses. All data is to be taken directly from the supplied tables; only listed options are considered.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, requires, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options: Each item_ref-location_id pair (from the union of all item tables; only those listed are modeled).
    - Resources: Each display section (e.g., SECTION_A, SECTION_B, SECTION_C).
    - Categories: Each product category (e.g., G0, G1, G2, G3).
    - Bundles: Each pair of options eligible for a bundle bonus.
    - Incompatibility pairs: Each pair of options that cannot be jointly selected.
    - Requires pairs: Each (option, prerequisite) pair where one option requires another.
4.  **Define Decision Variables:**
    -   `x[o]` = Number of packs to display for option o (item_ref-location_id). Type: GRB.INTEGER, domain: {0} or [minimum_lot, maximum_order] if authorized.
    -   `y[o]` = 1 if option o is selected (i.e., x[o] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any option in category c is selected (i.e., sum of x[o] for category c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both options in bundle are selected (i.e., both y[o_a] and y[o_b] = 1), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each option: sum of amount_cents for each item_ref in the 'benefit' table.
    -   Per-option activation fee: activation_fee_cents from 'item_fee' table, applied once if x[o] > 0.
    -   Per-category activation fee: activation_fee_cents from 'category' table, applied once if any x[o] > 0 for that category.
    -   Bundle bonuses: bonus_cents from 'bundle' table, applied once if both options are selected.
    -   Option authorization, minimum_lot, maximum_order, category, and location_id: from 'item' tables.
    -   Resource usage per option: amount (converted from liters to ml) from 'usage' tables, by resource.
    -   Section (resource) capacity: sum of 'amount' in 'capacity_ledger' table for each resource.
    -   Category quantity bounds: minimum_quantity and maximum_quantity from 'category' table.
    -   Incompatibility pairs: from 'incompatible' table.
    -   Requires pairs: from 'requires' table.
6.  **Formulate Objective:** Maximize total net merchandising benefit, defined as:
    -   Sum over all options of (per-unit benefit * x[o])
    -   Minus sum over all selected options of their item_fee (activation_fee_cents)
    -   Minus sum over all used categories of their activation_fee_cents
    -   Plus sum over all eligible bundles of their bonus_cents (if both options in the bundle are selected)
    -   All amounts are in USD cents.
7.  **Formulate Constraints:**
    -   **Option Authorization and Lot/Order Limits:** For each option o:
        - If authorized = 0, x[o] = 0.
        - If authorized = 1, x[o] ∈ {0} ∪ [minimum_lot, maximum_order], integer.
        - y[o] = 1 if x[o] > 0, else 0.
    -   **Resource (Section) Capacity:** For each resource r (section):
        - sum over all options assigned to r of (usage per pack in ml * x[o]) ≤ total available capacity for r (sum of 'amount' in 'capacity_ledger' for r).
    -   **Category Quantity Bounds:** For each category c:
        - sum over all options in c of x[o] ≥ minimum_quantity[c]
        - sum over all options in c of x[o] ≤ maximum_quantity[c]
        - z[c] = 1 if sum over x[o] in c > 0, else 0.
    -   **Incompatibility:** For each incompatible pair (o1, o2):
        - y[o1] + y[o2] ≤ 1 (cannot both be selected).
    -   **Requires:** For each (o, prereq):
        - y[o] ≤ y[prereq] (if o is selected, prereq must also be selected; no proportionality).
    -   **Bundle Bonuses:** For each bundle (o_a, o_b):
        - b[bundle] ≤ y[o_a]
        - b[bundle] ≤ y[o_b]
        - b[bundle] ≥ y[o_a] + y[o_b] - 1 (b[bundle] = 1 iff both options are selected).
    -   **Variable Linking:** For each option o:
        - x[o] ≥ minimum_lot[o] * y[o]
        - x[o] ≤ maximum_order[o] * y[o]
    -   **Domain:** All x[o] are integer, y[o], z[c], b[bundle] are binary.
[Abstract Model Plan END]