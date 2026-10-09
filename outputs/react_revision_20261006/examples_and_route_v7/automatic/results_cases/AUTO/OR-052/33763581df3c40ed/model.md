[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal integer allocation of multiple types of books to several bookshelves, maximizing the total value of books placed, while ensuring that the total weight of books on each bookshelf does not exceed its capacity.
2.  **Identify Model Type:** Based on the query, this is a Mixed Integer Programming (MIP) problem, specifically a multi-knapsack problem (multiple bins, integer quantities).
3.  **Define Index Sets:** The primary indices are:
    - Bookshelves (from `capacity.csv`, indexed by `BookshelfID`)
    - Books (from `products.csv`, indexed by `ProductName`)
4.  **Define Decision Variables:**
    -   `x[i,j]` = Number of units of book j placed on bookshelf i. Type: GRB.INTEGER (must be non-negative integers).
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'Value' column from `products.csv` (the value per unit of each book).
    -   Constraint coefficients: 'Weight' column from `products.csv` (the weight per unit of each book).
    -   Constraint RHS (limits): 'Capacity' column from `capacity.csv` (the maximum total weight allowed on each bookshelf).
6.  **Formulate Objective:** Maximize the total value of all books placed on all bookshelves, i.e., maximize the sum over all bookshelves and books of (value per unit of book j) × (number of units of book j placed on bookshelf i).
7.  **Formulate Constraints:**
    -   Constraint 1 (Bookshelf Capacity): For each bookshelf i, the sum over all books j of (weight per unit of book j) × (number of units of book j placed on bookshelf i) must be less than or equal to the capacity of bookshelf i.
    -   Constraint 2 (Non-negativity and Integrality): For all i and j, x[i,j] ≥ 0 and integer.
    -   (No explicit upper bound on the number of units per book per shelf is given, so only the capacity constraint applies unless further limits are specified.)
[Abstract Model Plan END]