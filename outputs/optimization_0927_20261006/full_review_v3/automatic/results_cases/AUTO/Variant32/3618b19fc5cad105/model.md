[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine, for the Oslo dealership as of 2026-05-07, the vehicle order (integer lot sizes for authorized options) that yields the largest net benefit in USD cents, accounting for multi-currency benefit components, item and category activation fees, bundle bonuses, resource and category limits, option incompatibilities, and requires dependencies, using only valid records per the specified selection rules.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge, resource, and logical (incompatibility/requirement) constraints.
3.  **Define Index Sets:** The primary indices are:
    - Options (items) \( i \) available for order.
    - Categories \( g \) to which options belong.
    - Resources \( r \) (e.g., space, power, labor).
    - Bundles \( b \) (pairs of options with a bonus).
    - Incompatible pairs \( (i, j) \).
    - Requires pairs \( (i, k) \).
4.  **Define Decision Variables:**
    - \( x_i \) = Integer quantity of option \( i \) ordered (must be 0 if unauthorized; otherwise between minimum_lot and maximum_order). Type: GRB.INTEGER.
    - \( y_i \) = 1 if option \( i \) is selected (i.e., \( x_i > 0 \)), 0 otherwise. Type: GRB.BINARY.
    - \( z_g \) = 1 if any option in category \( g \) is selected (i.e., category is used), 0 otherwise. Type: GRB.BINARY.
    - \( w_b \) = 1 if both options in bundle \( b \) are selected, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Benefit per unit (in USD cents) for each option: computed by summing all benefit components (from 'benefit' tables) for each item, converting each amount to USD cents using the corresponding 'fx' table (fields: 'amount', 'currency', 'usd_cents_numerator', 'denominator').
    - Item activation fee (USD cents): from 'item_fee' tables, field 'activation_fee_cents' per item_ref.
    - Category activation fee (USD cents): from 'category' tables, field 'activation_fee_cents' per category.
    - Bundle bonus (USD cents): from 'bundle' tables, field 'bonus_cents' per bundle (item_a, item_b).
    - Resource usage per unit: from 'usage' tables, fields 'item_ref', 'resource', 'amount', 'unit' (convert units to base: 1000 ml/liter, 60 min/hour, 1000 wh/kwh).
    - Resource capacity: from 'capacity_ledger' tables, sum of 'amount' for each resource (opening + reservations), in base units.
    - Option authorization, minimum_lot, maximum_order, and category: from 'item' tables, fields 'authorized', 'minimum_lot', 'maximum_order', 'category'.
    - Category minimum_quantity, maximum_quantity: from 'category' tables, fields 'minimum_quantity', 'maximum_quantity'.
    - Incompatible pairs: from 'incompatible' tables, fields 'item_a', 'item_b'.
    - Requires dependencies: from 'requires' tables, fields 'item_ref', 'prerequisite_ref'.
6.  **Formulate Objective:** Maximize total net benefit in USD cents, defined as:
    - Sum over all options: (benefit per unit in USD cents) × (quantity ordered)
    - Minus: sum of item activation fees for each option with positive quantity (charge once per option)
    - Minus: sum of category activation fees for each category with any option selected (charge once per category)
    - Plus: sum of bundle bonuses for each bundle where both options are selected (award once per bundle)
7.  **Formulate Constraints:**
    - **Option Authorization and Lot Size:** For each option \( i \):
        - If authorized: \( x_i = 0 \) or \( x_i \in [\text{minimum_lot}_i, \text{maximum_order}_i] \) (integer).
        - If unauthorized: \( x_i = 0 \).
        - \( y_i = 1 \) if \( x_i > 0 \), else 0.
    - **Category Quantity Limits:** For each category \( g \):
        - \( \sum_{i \in g} x_i \in [\text{minimum_quantity}_g, \text{maximum_quantity}_g] \) (unconditional).
        - \( z_g = 1 \) if any \( x_i > 0 \) for \( i \in g \), else 0.
    - **Resource Capacity:** For each resource \( r \):
        - \( \sum_{i} (\text{usage per unit of } i \text{ for } r) \times x_i \leq \text{signed total capacity}_r \) (all in base units).
    - **Item and Category Activation Fees:** Activation fees are charged once per item/category if used (modeled via \( y_i \), \( z_g \)).
    - **Incompatibility:** For each incompatible pair \( (i, j) \):
        - \( y_i + y_j \leq 1 \) (cannot select both).
    - **Requires Dependencies:** For each requires pair \( (i, k) \):
        - \( x_i > 0 \implies x_k > 0 \) (i.e., \( y_i \leq y_k \)).
    - **Bundle Bonuses:** For each bundle \( b = (i, j) \):
        - \( w_b = 1 \) if \( y_i = 1 \) and \( y_j = 1 \), else 0; bonus awarded only if both options are selected and authorized.
    - **Variable Domains:** All variables as defined above; all quantities integer; all binaries in {0,1}.
[Abstract Model Plan END]