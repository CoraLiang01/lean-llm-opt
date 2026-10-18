[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate a set of products to a set of display shelves so that the total value of the products placed on all shelves is maximized, while ensuring that the total weight of products on each shelf does not exceed that shelf’s capacity. The decision variable \(x_{ij}\) represents the number of units of product \(j\) placed on shelf \(i\).

2.  **Identify Model Type:** This is a linear programming (LP) problem (continuous variables). If the problem requires integer units of products, it can be treated as a mixed‑integer linear program (MILP) by declaring \(x_{ij}\) as integer.

3.  **Define Index Sets:**  
   - \(I\): set of shelves (indices from the `ShelfID` column of *capacity.csv*).  
   - \(J\): set of products (indices from the `ProductName` column of *products.csv*).

4.  **Define Decision Variables:**  
   - `x[i,j]` = number of units of product \(j\) placed on shelf \(i\).  
     Type: `GRB.CONTINUOUS` (use `GRB.INTEGER` if units must be whole numbers).  
     Bounds: \(x_{ij} \ge 0\).

5.  **Identify Parameters (from Schema):**  
   - `Capacity[i]` (from the `Capacity` column of *capacity.csv*).  
   - `Value[j]` (from the `Value` column of *products.csv*).  
   - `Weight[j]` (from the `Weight` column of *products.csv*).

6.  **Formulate Objective:**  
   Maximize total value of products on all shelves:  
   \[
   \max \sum_{i \in I}\sum_{j \in J} \text{Value}[j] \times x_{ij}
   \]

7.  **Formulate Constraints:**  
   - **Shelf capacity constraints** (for each shelf \(i\)):  
     \[
     \sum_{j \in J} \text{Weight}[j] \times x_{ij} \;\le\; \text{Capacity}[i]
     \]
   - **Non‑negativity** (for all \(i,j\)):  
     \[
     x_{ij} \;\ge\; 0
     \]
   (If integer units are required, add `x[i,j]` ∈ ℤ⁺.)

[Abstract Model Plan END]