[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal number of packs to display for each product-and-section option (item_ref, location_id) in MARKET_SQUARE, maximizing net merchandising benefit. The plan must respect section capacities, per-option authorization and order limits, resource usage, category quantity bounds, incompatibility and prerequisite requirements, and bundle bonuses. Fixed item and category activation fees are deducted if used.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (activation fee) and logical (incompatibility, prerequisite, bundle) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Item options: All item_ref-location_id pairs (from the union of batch_01/export_07.csv and batch_02/export_08.csv; use all rows as the query requests to use their rows directly).
    - Resources: Each display section (SECTION_A, SECTION_B, SECTION_C).
    - Categories: Each product category (G0, G1, G2, G3).
    - Bundles: Each bundle pair (from batch_02/export_02.csv).
    - Incompatibility pairs and prerequisite pairs (from batch_06/export_06.csv and batch_05/export_11.csv).
4.  **Define Decision Variables:**
    -   `x[i]` = Number of packs to display for item option i (item_ref-location_id). Type: GRB.INTEGER, with bounds [0, maximum_order_i], and x[i]=0 if not authorized.
    -   `y[i]` = 1 if item option i is selected (x[i] > 0), 0 otherwise. Type: GRB.BINARY.
    -   `z[c]` = 1 if any item in category c is selected (sum of x[i] for category c > 0), 0 otherwise. Type: GRB.BINARY.
    -   `b[bundle]` = 1 if both items in bundle are selected (x[item_a] > 0 and x[item_b] > 0), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Per-unit benefit for each item option: sum of amount_cents from all 'benefit' rows for that item_ref (batch_01/export_01.csv).
    -   Item activation fee: activation_fee_cents per item_ref (batch_03/export_09.csv).
    -   Category activation fee: activation_fee_cents per category (batch_04/export_04.csv).
    -   Bundle bonus: bonus_cents per bundle (batch_02/export_02.csv).
    -   Resource usage per item option: amount (converted from liters to ml) per item_ref-resource (batch_06/export_12.csv and batch_01/export_13.csv).
    -   Section (resource) capacity: sum of amount for each resource in capacity_ledger (batch_03/export_03.csv).
    -   Category bounds: minimum_quantity and maximum_quantity per category (batch_04/export_04.csv).
    -   Authorization, minimum_lot, maximum_order, category, location_id for each item option (batch_01/export_07.csv and batch_02/export_08.csv).
    -   Incompatibility pairs: item_a, item_b (batch_06/export_06.csv).
    -   Prerequisite pairs: item_ref, prerequisite_ref (batch_05/export_11.csv).
6.  **Formulate Objective:** Maximize total net merchandising benefit in USD cents:
    -   Sum over all item options: (per-unit benefit) × x[i]
    -   Minus sum over all selected item options: item activation_fee_cents × y[i]
    -   Minus sum over all used categories: category activation_fee_cents × z[c]
    -   Plus sum over all bundles: bonus_cents × b[bundle]
7.  **Formulate Constraints:**
    -   **Authorization and Order Bounds:** For each item option i, x[i] = 0 if not authorized; otherwise, minimum_lot_i ≤ x[i] ≤ maximum_order_i, x[i] integer.
    -   **Section (Resource) Capacity:** For each section/resource r, sum over all item options assigned to r of (resource usage per pack in ml) × x[i] ≤ total available capacity for r (sum of capacity_ledger entries for r).
    -   **Category Quantity Bounds:** For each category c, sum over all x[i] for item options in c must be between minimum_quantity_c and maximum_quantity_c.
    -   **Item Activation Indicator:** For each item option i, y[i] = 1 if x[i] > 0, 0 otherwise (enforced via y[i] ≥ x[i]/maximum_order_i and x[i] ≤ maximum_order_i × y[i]).
    -   **Category Activation Indicator:** For each category c, z[c] = 1 if any x[i] > 0 for i in c, 0 otherwise (z[c] ≥ x[i]/maximum_order_i for all i in c).
    -   **Bundle Bonus Indicator:** For each bundle (item_a, item_b), b[bundle] = 1 if both y[item_a] = 1 and y[item_b] = 1, 0 otherwise (b[bundle] ≤ y[item_a], b[bundle] ≤ y[item_b], b[bundle] ≥ y[item_a] + y[item_b] - 1).
    -   **Incompatibility:** For each incompatible pair (item_a, item_b), y[item_a] + y[item_b] ≤ 1.
    -   **Prerequisite:** For each requires pair (item_ref, prerequisite_ref), y[item_ref] ≤ y[prerequisite_ref].
    -   **Zero for Unauthorized:** For any item option with authorized = 0, x[i] = 0.
    -   **Integrality:** All x[i] are integer, y[i], z[c], b[bundle] are binary.
[Abstract Model Plan END]