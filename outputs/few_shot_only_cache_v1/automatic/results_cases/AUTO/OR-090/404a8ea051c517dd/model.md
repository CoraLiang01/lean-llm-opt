[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products (P1–P100), where each batch is 10 units, in order to maximize total profit. The plan must respect per-product upper demand limits (in units), and total available capacities for three shared resources (R1, R2, R3). All products and resources are explicitly enumerated in the data; unmet demand is allowed, but overproduction is not.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with resource constraints and batch-size integer variables.
3.  **Define Index Sets:** The primary indices are:
    - Products: \( i \in \{\text{P1}, \ldots, \text{P100}\} \)
    - Resources: \( r \in \{\text{R1}, \text{R2}, \text{R3}\} \)
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product \( i \) to produce/purchase. Type: GRB.INTEGER, \( x[i] \geq 0 \).
        - Each batch is exactly 10 units (from `batch_size_units`).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `profit_per_unit` (from factory_products_100.csv), batch size (`batch_size_units` = 10).
    -   Constraint coefficients: `r1_per_unit`, `r2_per_unit`, `r3_per_unit` (resource consumption per unit, from factory_products_100.csv).
    -   Constraint RHS (limits): `upper_demand_units` (per product, from factory_products_100.csv); `capacity` for each resource (from resources_capacities.csv).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches × batch size × profit per unit):
    - Maximize \( \sum_{i} x[i] \times \text{batch\_size\_units} \times \text{profit\_per\_unit}[i] \)
7.  **Formulate Constraints:**
    -   **Resource Capacity Constraints:** For each resource \( r \in \{\text{R1}, \text{R2}, \text{R3}\} \), the total resource consumed by all produced batches cannot exceed the available capacity:
        - \( \sum_{i} x[i] \times \text{batch\_size\_units} \times \text{r\_per\_unit}[i][r] \leq \text{capacity}[r] \)
        - Where \(\text{r\_per\_unit}[i][r]\) is the per-unit consumption of resource \( r \) by product \( i \).
    -   **Demand Upper Bound Constraints:** For each product \( i \), the total units produced cannot exceed its upper demand:
        - \( x[i] \times \text{batch\_size\_units} \leq \text{upper\_demand\_units}[i] \)
    -   **Non-negativity and Integrality:** For all products \( i \):
        - \( x[i] \geq 0 \), integer.
[Abstract Model Plan END]