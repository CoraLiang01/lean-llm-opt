[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the daily order quantity of each produce type so as to maximize the total benefit (value) while ensuring that the total weight of all ordered units does not exceed the overall inventory capacity. All decision variables must be integers.

2.  **Identify Model Type:** This is an Integer Linear Programming (ILP) problem – specifically a 0‑1/continuous knapsack formulation with integer decision variables.

3.  **Define Index Sets:**  
   - `Products` – the set of all produce types listed in *products.csv* (10 items: Spinach, Shiitake Mushrooms, Apples, Carrots, Basil, Potatoes, Green Beans, Blueberries, Oranges, Watermelons).

4.  **Define Decision Variables:**  
   - `x[p]` = number of units of product `p` to order daily.  
     Type: `GRB.INTEGER` (non‑negative integers).  
     Domain: `x[p] >= 0`.

5.  **Identify Parameters (from Schema):**  
   - Objective coefficients: `Value[p]` (from the `Value` column of *products.csv*).  
   - Constraint coefficients: `Weight[p]` (from the `Weight` column of *products.csv*).  
   - Constraint RHS: `Capacity` (from the single value in *capacity.csv*).

6.  **Formulate Objective:**  
   Maximize  
   \[
   \sum_{p \in Products} \text{Value}[p] \times x[p]
   \]

7.  **Formulate Constraints:**  
   - **Capacity Constraint:**  
     \[
     \sum_{p \in Products} \text{Weight}[p] \times x[p] \;\le\; \text{Capacity}
     \]
   - **Non‑negativity & Integer Constraint:**  
     \[
     x[p] \;\in\; \mathbb{Z}_{\ge 0} \quad \forall p \in Products
     \]
   (No additional linking or binary variables are required because any integer quantity of each product can be ordered.)

[Abstract Model Plan END]