[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many units of each bread type to order daily in order to maximize the total expected profit while staying within the bakery’s storage capacity. The decision variables must be integers.

2. **Identify Model Type:** This is a Mixed Integer Linear Programming (MIP) problem – an integer knapsack formulation.

3. **Define Index Sets:**  
   - **Products** = {i | i = 1,…,10} corresponding to the 10 bread types listed in *products.csv*.

4. **Define Decision Variables:**  
   - `x[i]` = number of units of product *i* to order each day.  
     *Type:* `GRB.INTEGER`, lower bound `0`.

5. **Identify Parameters (from Schema):**  
   - **Profit[i]** comes from the column **Value** in *products.csv*.  
   - **Weight[i]** comes from the column **Weight** in *products.csv*.  
   - **Capacity** comes from the single value in *capacity.csv*.

6. **Formulate Objective:**  
   - Maximize  
     \[
     \sum_{i \in \text{Products}} \text{Profit}[i] \times x[i]
     \]

7. **Formulate Constraints:**  
   - **Storage Capacity:**  
     \[
     \sum_{i \in \text{Products}} \text{Weight}[i] \times x[i] \;\le\; \text{Capacity}
     \]
   - **Non‑negativity:**  
     \[
     x[i] \;\ge\; 0 \quad \forall i \in \text{Products}
     \]
[Abstract Model Plan END]