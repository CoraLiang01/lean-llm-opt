[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine how many units of each product should be placed on each display shelf so that the total value of all displayed products is maximized, while ensuring that the total weight of products on any shelf does not exceed that shelf’s capacity. All decision variables must be integers.

2.  **Identify Model Type:** This is a **Mixed Integer Programming (MIP)** problem (integer variables, linear objective, linear constraints).

3.  **Define Index Sets:**  
    - **Shelves** \(I\): set of all shelf identifiers (from `capacity.csv`, e.g., \(\{1,2,\dots,10\}\)).  
    - **Products** \(J\): set of all product names (from `products.csv`, e.g., \(\{ \text{Smartphone}, \text{Laptop}, \dots\}\)).

4.  **Define Decision Variables:**  
    - `x[i,j]` = number of units of product \(j\) placed on shelf \(i\).  
      Type: **GRB.INTEGER** (non‑negative integers).  
    - (No binary or continuous auxiliary variables are required.)

5.  **Identify Parameters (from Schema):**  
    - **Value** \(v_j\): value of one unit of product \(j\) (column `Value` in `products.csv`).  
    - **Weight** \(w_j\): weight of one unit of product \(j\) (column `Weight` in `products.csv`).  
    - **Capacity** \(c_i\): maximum total weight allowed on shelf \(i\) (column `Capacity` in `capacity.csv`).  

6.  **Formulate Objective:**  
    Maximize the total value of displayed products:  
    \[
    \max \sum_{i \in I}\sum_{j \in J} v_j \, x_{i,j}
    \]

7.  **Formulate Constraints:**  
    - **Shelf Capacity Constraints** (for each shelf \(i\)):  
      \[
      \sum_{j \in J} w_j \, x_{i,j} \;\le\; c_i
      \]
    - **Non‑negativity / Integer Constraints** (for all \(i,j\)):  
      \[
      x_{i,j} \;\ge\; 0 \quad \text{and integer}
      \]
    - (No additional constraints such as demand limits or inventory bounds are specified in the query, so they are omitted.)

[Abstract Model Plan END]