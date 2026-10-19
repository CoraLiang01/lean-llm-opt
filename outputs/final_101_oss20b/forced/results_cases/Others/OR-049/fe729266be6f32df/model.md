[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each product should be placed on each display shelf so that the total value of all products on all shelves is maximized, while ensuring that the total weight of the products on each shelf does not exceed that shelf’s capacity. All decision variables must be integers.

2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem (integer decision variables, linear objective and constraints).

3.  **Define Index Sets:**  
   - **I** = set of shelves (indices from the `capacity.csv` file).  
   - **J** = set of products (indices from the `products.csv` file).

4.  **Define Decision Variables:**  
   - `x[i,j]` = number of units of product *j* placed on shelf *i*.  
     Type: **GRB.INTEGER** (non‑negative integers).  

5.  **Identify Parameters (from Schema):**  
   - `Capacity[i]` – weight capacity of shelf *i* (from `capacity.csv`, column `Capacity`).  
   - `Weight[j]` – weight of one unit of product *j* (from `products.csv`, column `Weight`).  
   - `Value[j]` – monetary value of one unit of product *j* (from `products.csv`, column `Value`).  

6.  **Formulate Objective:**  
   Maximize the total value of all products placed on all shelves:  
   \[
   \max \sum_{i \in I}\sum_{j \in J} \text{Value}[j] \times x[i,j]
   \]

7.  **Formulate Constraints:**  
   - **Shelf Capacity Constraints** (for each shelf *i*):  
     \[
     \sum_{j \in J} \text{Weight}[j] \times x[i,j] \;\le\; \text{Capacity}[i]
     \]
   - **Non‑negativity and Integer Constraints** (for all *i*, *j*):  
     \[
     x[i,j] \;\ge\; 0 \quad \text{and integer}
     \]
[Abstract Model Plan END]