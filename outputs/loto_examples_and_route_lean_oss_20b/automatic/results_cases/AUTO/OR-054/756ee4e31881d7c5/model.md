[Abstract Model Plan START]  
1. **Analyze Query:** The user wants to allocate various types of products to different display shelves in order to maximize the total value of products placed on all shelves while respecting each shelf’s capacity limit.  
2. **Identify Model Type:** This is a **Linear Programming (LP)** problem (can be treated as a Mixed‑Integer Program if integer units are required).  
3. **Define Index Sets:**  
   - **I** – set of shelves (ShelfID values from *capacity.csv*).  
   - **J** – set of products (ProductName values from *products.csv*).  
4. **Define Decision Variables:**  
   - `x[i,j]` = number of units of product j placed on shelf i.  
     *Type:* `GRB.CONTINUOUS` (use `GRB.INTEGER` if units must be whole).  
5. **Identify Parameters (from Schema):**  
   - **Capacity[i]** – shelf capacity from *capacity.csv* (`Capacity`).  
   - **Value[j]** – product value from *products.csv* (`Value`).  
   - **Weight[j]** – product weight from *products.csv* (`Weight`).  
6. **Formulate Objective:**  
   - Maximize total value:  
     \[
     \max \sum_{i\in I}\sum_{j\in J} \text{Value}[j] \; x[i,j]
     \]  
7. **Formulate Constraints:**  
   - **Shelf capacity constraints (for each shelf i):**  
     \[
     \sum_{j\in J} \text{Weight}[j] \; x[i,j] \;\le\; \text{Capacity}[i]
     \]  
   - **Non‑negativity (for all i, j):**  
     \[
     x[i,j] \;\ge\; 0
     \]  
   - *(Optional integer constraint if required)*:  
     \[
     x[i,j] \;\in\; \mathbb{Z}_{\ge0}
     \]  
[Abstract Model Plan END]