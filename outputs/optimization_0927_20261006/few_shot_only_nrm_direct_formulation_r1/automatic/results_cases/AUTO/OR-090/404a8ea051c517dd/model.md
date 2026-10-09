[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce/purchase for each of 100 products, where each batch is 10 units, in order to maximize total profit. The plan must respect per-product upper demand limits (in units), and aggregate resource consumption limits for three resources (R1, R2, R3). Unmet demand is allowed (i.e., not all demand must be satisfied), but overproduction beyond the upper demand is not allowed. All decisions are for a single planning horizon.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch variables, linear constraints).
3.  **Define Index Sets:** The primary indices are:
    - Products: \( i \in \text{Products} \) (from all 100 rows of `factory_products_100.csv`)
    - Resources: \( r \in \{\text{R1}, \text{R2}, \text{R3}\} \) (from all 3 rows of `resources_capacities.csv`)
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product \( i \) to produce/purchase. Type: GRB.INTEGER, \( x[i] \geq 0 \).
        - Each batch is exactly 10 units (from `batch_size_units`).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `profit_per_unit` (from `factory_products_100.csv`), batch size (`batch_size_units` = 10).
    -   Constraint coefficients: `r1_per_unit`, `r2_per_unit`, `r3_per_unit` (resource usage per unit, from `factory_products_100.csv`).
    -   Constraint RHS (limits): `upper_demand_units` (per product, from `factory_products_100.csv`), and `capacity` for each resource (from `resources_capacities.csv`).
6.  **Formulate Objective:** Maximize total profit, calculated as the sum over all products of (number of batches × batch size × profit per unit):  
    \[
    \text{Maximize} \quad \sum_{i \in \text{Products}} x[i] \times \text{batch\_size\_units} \times \text{profit\_per\_unit}[i]
    \]
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource \( r \in \{\text{R1}, \text{R2}, \text{R3}\} \), the total resource consumption across all products cannot exceed the available capacity:
        \[
        \sum_{i \in \text{Products}} x[i] \times \text{batch\_size\_units} \times \text{r\_per\_unit}[i][r] \leq \text{capacity}[r]
        \]
        where `r_per_unit[i][r]` is the per-unit usage of resource \( r \) for product \( i \).
    -   Constraint 2 (Demand Upper Bound): For each product \( i \), the total produced units cannot exceed the product's upper demand:
        \[
        x[i] \times \text{batch\_size\_units} \leq \text{upper\_demand\_units}[i]
        \]
        (Since `upper_demand_units` may not be a multiple of 10, this may force \( x[i] \) to be the largest integer such that \( x[i] \times 10 \leq \text{upper\_demand\_units}[i] \).)
    -   Constraint 3 (Non-negativity and Integrality): For all products \( i \), \( x[i] \geq 0 \) and integer.
[Abstract Model Plan END]