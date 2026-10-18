[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate various air‑conditioner types to multiple warehouse storage areas so that the total value of the stored units is maximized while respecting the size (weight) capacity of each storage area. The decision variables are the integer counts of each product type placed in each storage area.

2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem (integer variables, linear objective, linear constraints).

3.  **Define Index Sets:**  
   - **I** = set of storage area identifiers (from `capacity.csv`, values 1…15).  
   - **J** = set of product names (from `products.csv`, 10 distinct air‑conditioner types).

4.  **Define Decision Variables:**  
   - `x[i,j]` = number of units of product *j* placed in storage area *i*.  
     Type: **GRB.INTEGER** (non‑negative integers).  
   - No binary variables are required for this problem.

5.  **Identify Parameters (from Schema):**  
   - `Capacity[i]` – the maximum total weight that storage area *i* can hold (from `capacity.csv`).  
   - `Weight[j]` – the weight of one unit of product *j* (from `products.csv`).  
   - `Value[j]` – the monetary value of one unit of product *j* (from `products.csv`).  

6.  **Formulate Objective:**  
   Maximize the total value of all stored units:  
   \[
   \max \sum_{i \in I}\sum_{j \in J} \text{Value}[j] \times x[i,j]
   \]

7.  **Formulate Constraints:**  
   - **Capacity Constraints (for each storage area):**  
     \[
     \sum_{j \in J} \text{Weight}[j] \times x[i,j] \;\le\; \text{Capacity}[i] \quad \forall i \in I
     \]
   - **Non‑negativity and Integer Constraints:**  
     \[
     x[i,j] \;\ge\; 0 \quad \text{and integer} \quad \forall i \in I,\; j \in J
     \]
[Abstract Model Plan END]