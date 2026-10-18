[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to decide how many units of each product to place on each shelf so that the total value of all products on all shelves is maximised, while ensuring that the total weight of the products on each shelf does not exceed that shelf’s capacity. All decision variables must be integers.

2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem – specifically a multiple‑knapsack problem where each shelf is a knapsack with a weight capacity and each product has a value and a weight.

3.  **Define Index Sets:**  
   - **I** – set of shelves (ShelfID values from `capacity.csv`, 10 elements).  
   - **J** – set of products (ProductName values from `products.csv`, 20 elements).

4.  **Define Decision Variables:**  
   - `x[i,j]` = number of units of product *j* placed on shelf *i*.  
     Type: **GRB.INTEGER** (non‑negative integers).  
   - No binary or continuous auxiliary variables are required.

5.  **Identify Parameters (from Schema):**  
   - **Capacity[i]** – weight capacity of shelf *i* (column `Capacity` in `capacity.csv`).  
   - **Weight[j]** – weight of one unit of product *j* (column `Weight` in `products.csv`).  
   - **Value[j]** – monetary value of one unit of product *j* (column `Value` in `products.csv`).  

6.  **Formulate Objective:**  
   Maximise total value across all shelves:  
   \[
   \max \sum_{i \in I}\sum_{j \in J} \text{Value}[j] \times x[i,j]
   \]

7.  **Formulate Constraints:**  
   - **Weight capacity for each shelf:**  
     \[
     \forall i \in I:\quad \sum_{j \in J} \text{Weight}[j] \times x[i,j] \;\le\; \text{Capacity}[i]
     \]
   - **Non‑negativity and integrality of decision variables:**  
     \[
     \forall i \in I,\; \forall j \in J:\quad x[i,j] \;\ge\; 0,\; x[i,j] \in \mathbb{Z}
     \]
   - (No other constraints are required because the problem statement does not impose limits on product availability or shelf usage.)

[Abstract Model Plan END]