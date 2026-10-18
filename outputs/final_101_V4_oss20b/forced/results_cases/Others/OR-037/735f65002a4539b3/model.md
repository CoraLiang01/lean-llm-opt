[Abstract Model Plan START]
1. **Analyze Query:** The user wants to determine the daily ordering quantities for each vehicle model so that total profit is maximised while the total weight of all ordered vehicles does not exceed the available inventory capacity.

2. **Identify Model Type:** This is a Mixed‑Integer Linear Programming (MIP) problem – a knapsack‑type optimisation with integer decision variables.

3. **Define Index Sets:**  
   - **Products** = {i | i = 1,…,25} – one index for each row in *products.csv*.

4. **Define Decision Variables:**  
   - `x[i]` = number of vehicles of product *i* to order per day.  
     *Type:* `GRB.INTEGER` (non‑negative).

5. **Identify Parameters (from Schema):**  
   - **Profit[i]** – from column **Value** in *products.csv*.  
   - **Weight[i]** – from column **Weight** in *products.csv*.  
   - **Capacity** – from column **Capacity** in *capacity.csv* (single value 765).

6. **Formulate Objective:**  
   - **Maximise**  
     \[
     \sum_{i \in \text{Products}} \text{Profit}[i] \times x[i]
     \]

7. **Formulate Constraints:**  
   - **Capacity Constraint**  
     \[
     \sum_{i \in \text{Products}} \text{Weight}[i] \times x[i] \;\le\; \text{Capacity}
     \]
   - **Non‑negativity**  
     \[
     x[i] \;\ge\; 0 \quad \forall i \in \text{Products}
     \]
[Abstract Model Plan END]