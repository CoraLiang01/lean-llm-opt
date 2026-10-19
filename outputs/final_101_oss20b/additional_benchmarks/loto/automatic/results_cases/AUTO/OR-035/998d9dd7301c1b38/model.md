[Abstract Model Plan START]
1. **Analyze Query:** The user wants to decide how many units of each bread type to order each day so that the total expected profit is maximized while the total weight of the ordered bread does not exceed the bakery’s storage capacity.  
2. **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem (integer decision variables, linear objective and constraints).  
3. **Define Index Sets:**  
   - **Products** – the set of bread types listed in *products.csv* (10 items).  
4. **Define Decision Variables:**  
   - `x[i]` = number of units of product *i* to order that day.  
     *Type:* `GRB.INTEGER` (non‑negative integers).  
5. **Identify Parameters (from Schema):**  
   - **Objective coefficients:** `Value[i]` from *products.csv* (expected profit per unit).  
   - **Constraint coefficients:** `Weight[i]` from *products.csv* (weight per unit).  
   - **Constraint RHS:** `Capacity` from *capacity.csv* (total available storage weight).  
6. **Formulate Objective:**  
   - **Maximize**  
     \[
     \sum_{i \in \text{Products}} \text{Value}[i] \times x[i]
     \]  
     (total expected profit).  
7. **Formulate Constraints:**  
   - **Storage capacity:**  
     \[
     \sum_{i \in \text{Products}} \text{Weight}[i] \times x[i] \;\le\; \text{Capacity}
     \]  
   - **Non‑negativity & integrality:**  
     \[
     x[i] \;\ge\; 0 \quad \text{and integer for all } i \in \text{Products}
     \]  
[Abstract Model Plan END]