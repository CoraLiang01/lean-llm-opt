[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many real‑estate units to develop in each New York area in order to maximize the total development benefit while staying within an overall development capacity limit.  
2. **Identify Model Type:** This is a **linear programming (LP)** problem: continuous decision variables, a single linear objective, and a single linear resource constraint.  
3. **Define Index Sets:**  
   - **I** = {Queens, Brooklyn, Manhattan, Bronx, Staten Island, Harlem, Upper East Side, Lower Manhattan, Midtown, Long Island City, …} – all 20 area names listed in *products.csv*.  
4. **Define Decision Variables:**  
   - `x[i]` = scale of development per day in area *i*.  
     - Type: **GRB.CONTINUOUS**  
     - Domain: `x[i] ≥ 0` for all *i* ∈ I.  
5. **Identify Parameters (from Schema):**  
   - **Benefit coefficient**: `Value[i]` – the benefit per unit of development in area *i* (column *Value* in *products.csv*).  
   - **Resource consumption coefficient**: `Weight[i]` – the development capacity consumed per unit in area *i* (column *Weight* in *products.csv*).  
   - **Capacity limit**: `Capacity` – the total available development capacity (single value from column *Capacity* in *capacity.csv*).  
6. **Formulate Objective:**  
   - Maximize total benefit:  
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