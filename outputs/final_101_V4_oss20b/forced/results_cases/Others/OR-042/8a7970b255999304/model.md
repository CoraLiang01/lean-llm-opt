[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the daily order quantity of each drug type so that the total benefit is maximized while the total weight of all ordered units does not exceed the pharmacy’s overall inventory capacity. All decision variables must be integers.

2.  **Identify Model Type:** This is an Integer Linear Programming (ILP) problem – a classic unbounded knapsack formulation.

3.  **Define Index Sets:**  
   - `Products` – the set of all drug types listed in *products.csv* (20 items).

4.  **Define Decision Variables:**  
   - `x[p]` = number of units of product `p` to order daily.  
     Type: `GRB.INTEGER` (non‑negative integers).  

5.  **Identify Parameters (from Schema):**  
   - **Objective coefficients**: `Value[p]` from the `Value` column of *products.csv*.  
   - **Constraint coefficients**: `Weight[p]` from the `Weight` column of *products.csv*.  
   - **Constraint RHS**: `Capacity` from the single row of *capacity.csv* (value 4120).

6.  **Formulate Objective:**  
   - Maximize  
     \[
     \sum_{p \in Products} \text{Value}[p] \times x[p]
     \]

7.  **Formulate Constraints:**  
   - **Capacity constraint**:  
     \[
     \sum_{p \in Products} \text{Weight}[p] \times x[p] \;\le\; \text{Capacity}
     \]  
   - **Non‑negativity and integrality**:  
     \[
     x[p] \;\ge\; 0 \quad \text{and integer for all } p \in Products
     \]
[Abstract Model Plan END]