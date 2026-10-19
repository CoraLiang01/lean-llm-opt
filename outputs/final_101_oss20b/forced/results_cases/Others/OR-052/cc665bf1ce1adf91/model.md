[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate a set of book types to a set of bookshelves so that the total value of books placed on all shelves is maximized, while ensuring that the total weight of books on each individual shelf does not exceed that shelf’s capacity. The decision variables are the integer counts of each book type placed on each shelf.

2.  **Identify Model Type:** This is a Mixed‑Integer Programming (MIP) problem – specifically a multi‑dimensional knapsack/assignment problem with integer decision variables.

3.  **Define Index Sets:**  
   - **I** = {1, 2, … 10} – the set of BookshelfIDs from *capacity.csv*.  
   - **J** = {1, 2, … 25} – the set of ProductNames (book types) from *products.csv*.

4.  **Define Decision Variables:**  
    - `x[i,j]` = number of units of book type *j* placed on bookshelf *i*.  
      Type: **GRB.INTEGER** (non‑negative integers).  

5.  **Identify Parameters (from Schema):**  
    - **Capacity**: `capacity[i]` – the weight capacity of bookshelf *i* (from *capacity.csv*).  
    - **Weight**: `weight[j]` – the weight of one unit of book type *j* (from *products.csv*).  
    - **Value**: `value[j]` – the monetary value of one unit of book type *j* (from *products.csv*).  

6.  **Formulate Objective:**  
    Maximize the total value of books placed on all shelves:  
    \[
    \max \sum_{i \in I} \sum_{j \in J} \text{value}[j] \times x[i,j]
    \]

7.  **Formulate Constraints:**  
    - **Shelf Capacity Constraints** (for each shelf *i*):  
      \[
      \sum_{j \in J} \text{weight}[j] \times x[i,j] \;\le\; \text{capacity}[i]
      \]
    - **Non‑negativity and Integrality** (for all *i*, *j*):  
      \[
      x[i,j] \;\ge\; 0 \quad \text{and integer}
      \]
    - *(Optional)* If book supply limits were provided, add:  
      \[
      \sum_{i \in I} x[i,j] \;\le\; \text{Supply}[j]
      \]
      but no such limits are specified in the current data.

[Abstract Model Plan END]