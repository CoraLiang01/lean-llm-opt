[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products, where each batch is 10 units, in order to maximize total profit. The plan must respect three resource capacity constraints and ensure that no product’s total production exceeds its upper demand bound (unmet demand is allowed).
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary indices are:
    - Products: \( i \in \{\text{P1}, \text{P2}, ..., \text{P100}\} \)
    - Resources: \( r \in \{\text{R1}, \text{R2}, \text{R3}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product \( i \) to produce/purchase. Type: GRB.INTEGER (non-negative).
        - Each batch is exactly 10 units, so total units for product \( i \) is \( 10 \times x[i] \).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients:
        - `profit_per_unit` (from factory_products_100.csv): profit earned per produced unit for each product.
        - `batch_size_units` (from factory_products_100.csv): always 10, the number of units per batch.
    -   Constraint coefficients:
        - `r1_per_unit`, `r2_per_unit`, `r3_per_unit` (from factory_products_100.csv): resource consumption per unit for each product and resource.
    -   Constraint RHS (limits):
        - `upper_demand_units` (from factory_products_100.csv): maximum allowed units for each product.
        - `capacity` (from resources_capacities.csv): total available amount for each resource (R1, R2, R3).
6.  **Formulate Objective:** Maximize total profit across all products, i.e., maximize the sum over all products of (number of batches) × (batch size) × (profit per unit):
    - Objective: Maximize \( \sum_{i} 10 \times x[i] \times \text{profit\_per\_unit}_i \)
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource \( r \in \{\text{R1}, \text{R2}, \text{R3}\} \), the total resource consumption across all products cannot exceed the available capacity:
        - \( \sum_{i} 10 \times x[i] \times \text{rX\_per\_unit}_i \leq \text{capacity}_r \)
        - Where rX is r1, r2, or r3 for each resource.
    -   Constraint 2 (Demand Upper Bound): For each product \( i \), the total produced units cannot exceed the product’s upper demand:
        - \( 10 \times x[i] \leq \text{upper\_demand\_units}_i \)
        - Since batch size is 10, this ensures only feasible batch counts are chosen.
    -   Constraint 3 (Non-negativity and Integrality): For each product \( i \):
        - \( x[i] \geq 0 \), integer.
[Abstract Model Plan END]