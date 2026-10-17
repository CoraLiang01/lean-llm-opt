[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer number of batches to produce for each of 100 products (P1–P100), where each batch is 10 units, in order to maximize total profit. Production is limited by three resource capacities and by each product’s upper demand (in units). Unmet demand is allowed, but overproduction is not. All decisions must be in whole batches.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem (integer batch decisions, linear constraints).
3.  **Define Index Sets:** The primary index is Products, denoted as $i \in \{\text{P1}, \ldots, \text{P100}\}$.
4.  **Define Decision Variables:**
    -   `x[i]` = Number of batches of product $i$ to produce (each batch = 10 units). Type: GRB.INTEGER, $x[i] \geq 0$.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: `profit_per_unit` (from factory_products_100.csv), batch size (`batch_size_units` = 10).
    -   Constraint coefficients: `r1_per_unit`, `r2_per_unit`, `r3_per_unit` (resource usage per unit, from factory_products_100.csv).
    -   Constraint RHS: `capacity` for R1, R2, R3 (from resources_capacities.csv); `upper_demand_units` for each product (from factory_products_100.csv).
6.  **Formulate Objective:** Maximize total profit, i.e., maximize $\sum_{i} \left( \text{batch\_size\_units} \times x[i] \times \text{profit\_per\_unit}[i] \right)$.
7.  **Formulate Constraints:**
    -   Constraint 1 (Resource Limits): For each resource $r \in \{\text{R1}, \text{R2}, \text{R3}\}$, $\sum_{i} \left( \text{batch\_size\_units} \times x[i] \times \text{r}_r\_\text{per\_unit}[i] \right) \leq \text{capacity}[r]$.
    -   Constraint 2 (Demand Upper Bound): For each product $i$, $x[i] \times \text{batch\_size\_units} \leq \text{upper\_demand\_units}[i]$.
    -   Constraint 3 (Batch Integrality and Non-negativity): For each product $i$, $x[i] \in \mathbb{Z}_{\geq 0}$.
[Abstract Model Plan END]