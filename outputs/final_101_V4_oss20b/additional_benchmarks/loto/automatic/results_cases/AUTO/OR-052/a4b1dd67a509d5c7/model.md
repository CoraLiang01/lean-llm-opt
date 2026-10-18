[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to allocate a set of book types to a set of bookshelves so that the total value of books placed on all shelves is maximized, while respecting the weight capacity of each shelf. The allocation variables must be integer counts of books.
2.  **Identify Model Type:** This is a **Mixed Integer Linear Programming (MIP)** problem (integer decision variables, linear objective and constraints).
3.  **Define Index Sets:**  
    - **I** = {1, 2, …, 10} – the set of bookshelves (from *capacity.csv*).  
    - **J** = {1, 2, …, 25} – the set of book types (from *products.csv*).  
4.  **Define Decision Variables:**  
    - `x[i,j]` = number of units of book type *j* placed on bookshelf *i*.  
      Type: **GRB.INTEGER** (non‑negative).  
5.  **Identify Parameters (from Schema):**  
    - `Capacity[i]` – weight capacity of bookshelf *i* (column *Capacity* in *capacity.csv*).  
    - `Weight[j]` – weight of one unit of book *j* (column *Weight* in *products.csv*).  
    - `Value[j]` – value of one unit of book *j* (column *Value* in *products.csv*).  
6.  **Formulate Objective:**  
    - **Maximize**  
      \[
      \sum_{i \in I}\;\sum_{j \in J}\; \text{Value}[j] \times x[i,j]
      \]
      (total value of all books placed on all shelves).
7.  **Formulate Constraints:**  
    - **Capacity Constraints (per shelf):**  
      \[
      \sum_{j \in J}\; \text{Weight}[j] \times x[i,j] \;\le\; \text{Capacity}[i] \quad \forall i \in I
      \]
    - **Non‑negativity and Integer Constraints:**  
      \[
      x[i,j] \;\ge\; 0 \quad \text{and integer} \quad \forall i \in I,\; j \in J
      \]
[Abstract Model Plan END]