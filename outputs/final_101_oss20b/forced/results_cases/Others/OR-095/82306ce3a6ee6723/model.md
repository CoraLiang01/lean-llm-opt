[Abstract Model Plan START]  
1. **Analyze Query:** The user wants to determine the optimal monthly production quantities for 141 aerospace‑grade widgets (Widget1 … Widget141) and the amount of CatalystX (a by‑product from Widget3) to sell, so as to maximize total profit while respecting monthly labor, material, and CatalystX market constraints.  
2. **Identify Model Type:** This is a **Linear Programming (LP)** problem – all decision variables are continuous, the objective is linear, and all constraints are linear.  
3. **Define Index Sets:**  
   - **Products** \(P = \{ \text{Widget1}, \text{Widget2}, \dots, \text{Widget141} \}\)  
   - **CatalystX** is a by‑product produced only from **Widget3**.  
4. **Define Decision Variables:**  
   - `x[i]` = quantity of product *i* produced (kg or units, as defined in the data). **Type:** `GRB.CONTINUOUS`.  
   - `s` = kilograms of CatalystX sold to the market. **Type:** `GRB.CONTINUOUS`.  
   - `u` = kilograms of CatalystX disposed as hazardous waste. **Type:** `GRB.CONTINUOUS`.  
5. **Identify Parameters (from Schema):**  
   - **Profit per unit**: `Profit[i]` from `product_resources.csv` column *Profit*.  
   - **Resource usage per unit**:  
     - `LaborHours[i]` from column *LaborHours*.  
     - `MaterialA[i]` from column *MaterialA*.  
     - `MaterialB[i]` from column *MaterialB*.  
   - **CatalystX production rate**: 5 kg per unit of **Widget3**.  
   - **Resource limits** (from `resource_limits.csv`):  
     - `LaborHoursLimit` = 5 000.  
     - `MaterialALimit` = 24 000.  
     - `MaterialBLimit` = 15 000.  
   - **CatalystX market parameters**:  
     - Sale price = \$300 / kg.  
     - Disposal cost = \$200 / kg.  
     - Monthly sale cap = 1 500 kg.  
6. **Formulate Objective:**  
   \[
   \max \;\; \sum_{i \in P} \text{Profit}[i] \cdot x[i] \;+\; 300 \cdot s \;-\; 200 \cdot u
   \]
   (Using the equality \(u = 5\,x[\text{Widget3}] - s\) this can be rewritten as  
   \(\max \sum_{i} \text{Profit}[i]\,x[i] + 500\,s - 1000\,x[\text{Widget3}]\).)  
7. **Formulate Constraints:**  
   - **Labor constraint**  
     \[
     \sum_{i \in P} \text{LaborHours}[i] \cdot x[i] \;\le\; 5\,000
     \]  
   - **Material A constraint**  
     \[
     \sum_{i \in P} \text{MaterialA}[i] \cdot x[i] \;\le\; 24\,000
     \]  
   - **Material B constraint**  
     \[
     \sum_{i \in P} \text{MaterialB}[i] \cdot x[i] \;\le\; 15\,000
     \]  
   - **CatalystX production–sale linkage**  
     \[
     u \;=\; 5 \cdot x[\text{Widget3}] \;-\; s
     \]  
   - **CatalystX sale cap**  
     \[
     s \;\le\; 1\,500
     \]  
   - **Non‑negativity**  
     \[
     x[i] \;\ge\; 0 \;\;\forall i \in P,\qquad s \;\ge\; 0,\qquad u \;\ge\; 0
     \]  
[Abstract Model Plan END]