[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many batches of each of 100 products to produce in a single planning horizon, maximizing total profit while respecting resource capacities and demand limits. Production must be in whole batches of 10 units, so decision variables are integer batch counts.

2. **Identify Model Type:** This is a Mixed Integer Linear Programming (MIP) problem (integer variables, linear objective and constraints).

3. **Define Index Sets:**  
   - Products = {P1, P2, …, P100}  
   - Resources = {R1, R2, R3}

4. **Define Decision Variables:**  
   - `x[i]` = number of batches of product *i* to produce. Type: `GRB.INTEGER` (non‑negative).

5. **Identify Parameters (from Schema):**  
   - `profit_per_unit[i]` from *factory_products_100.csv*.  
   - `r1_per_unit[i]`, `r2_per_unit[i]`, `r3_per_unit[i]` from *factory_products_100.csv*.  
   - `upper_demand_units[i]` from *factory_products_100.csv*.  
   - `batch_size_units[i]` (constant 10) from *factory_products_100.csv*.  
   - `capacity[R]` from *resources_capacities.csv*.

6. **Formulate Objective:**  
   Maximize  
   \[
   \sum_{i\in \text{Products}} 10 \times \text{profit\_per\_unit}[i] \times x[i]
   \]

7. **Formulate Constraints:**  
   - **Resource constraints**  
     \[
     \sum_{i} 10 \times \text{r1\_per\_unit}[i] \times x[i] \;\le\; \text{capacity}[R1]
     \]
     \[
     \sum_{i} 10 \times \text{r2\_per\_unit}[i] \times x[i] \;\le\; \text{capacity}[R2]
     \]
     \[
     \sum_{i} 10 \times \text{r3\_per\_unit}[i] \times x[i] \;\le\; \text{capacity}[R3]
     \]
   - **Demand constraints**  
     \[
     10 \times x[i] \;\le\; \text{upper\_demand\_units}[i] \quad \forall i\in \text{Products}
     \]
   - **Non‑negativity and integrality**  
     \[
     x[i] \;\ge\; 0,\; x[i]\in \mathbb{Z} \quad \forall i\in \text{Products}
     \]
[Abstract Model Plan END]