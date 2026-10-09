[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of multiple types of books to several bookshelves, maximizing the total value of books placed, while ensuring that the total weight of books on each bookshelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem (specifically, a multi-knapsack problem with integer variables).
3.  **Define Index Sets:** The primary indices are:
    - Bookshelves (indexed by i, from 'BookshelfID' in capacity.csv)
    - Books (indexed by j, from 'ProductName' in products.csv)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of book j placed on bookshelf i. Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from products.csv (value per unit of book j).
    -   Constraint coefficients: 'Weight' column from products.csv (weight per unit of book j).
    -   Constraint RHS (limits): 'Capacity' column from capacity.csv (maximum total weight per bookshelf i).
6.  **Formulate Objective:** Maximize the total value of all books placed on all bookshelves, i.e., maximize the sum over all bookshelves and books of (Value of book j) × (number of units of book j placed on bookshelf i).
7.  **Formulate Constraints:**
    -   Constraint 1 (Bookshelf Capacity): For each bookshelf i, the sum over all books j of (Weight of book j) × x[i,j] ≤ Capacity of bookshelf i.
    -   Constraint 2 (Non-negativity and Integrality): For all i and j, x[i,j] ≥ 0 and integer.
[Abstract Model Plan END]