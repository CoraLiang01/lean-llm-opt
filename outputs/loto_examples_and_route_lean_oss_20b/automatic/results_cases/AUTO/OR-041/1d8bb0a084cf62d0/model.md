[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many real‑estate units to develop in each area (Queens, Brooklyn, etc.) in order to maximize the total development benefit while staying within an overall development capacity limit.

2. **Identify Model Type:** This is a **Linear Programming (LP)** problem (continuous decision variables, linear objective and constraints).

3. **Define Index Sets:**  
   - **I** = {i | i is an area listed in `products.csv`}  
     (20 areas: Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, …)

4. **Define Decision Variables:**  
   - `x[i]` = scale of development per day in area *i*.  
     Type: `GRB.CONTINUOUS`, lower bound `0`.

5. **Identify Parameters (from Schema):**  
   - `Value[i]` – benefit per unit of development in area *i* (column **Value** in `products.csv`).  
   - `Weight[i]` – development effort (or “weight”) per unit in area *i* (column **Weight** in `products.csv`).  
   - `Capacity` – total available development capacity per day (column **Capacity** in `capacity.csv`, value `586`).

6. **Formulate Objective:**  
   Maximize total benefit:  
   \[
   \max \sum_{i \in I} \text{Value}[i] \times x[i]
   \]

7. **Formulate Constraints:**  
   - **Capacity Constraint:**  
     \[
     \sum_{i \in I} \text{Weight}[i] \times x[i] \;\le\; \text{Capacity}
     \]
   - **Non‑negativity:**  
     \[
     x[i] \;\ge\; 0 \quad \forall i \in I
     \]
[Abstract Model Plan END]