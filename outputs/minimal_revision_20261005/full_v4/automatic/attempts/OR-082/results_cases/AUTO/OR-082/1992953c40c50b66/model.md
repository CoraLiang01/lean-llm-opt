[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal sequence for a delivery van to visit three customer locations (A, B, and C) exactly once each, starting and ending at the depot, so as to minimize the total travel distance. The pairwise distances between all locations (including the depot) are provided in DistanceMatrix.csv.
2.  **Identify Model Type:** Based on the query, this is a Traveling Salesman Problem (TSP), which is a combinatorial optimization problem typically formulated as a Mixed Integer Program (MIP).
3.  **Define Index Sets:** The primary indices are the set of locations to be visited: {Depot, A, B, C}. (Other locations in the CSV are not relevant, as the query specifies only these four.)
4.  **Define Decision Variables:**
    -   `x[i, j]` = 1 if the route travels directly from location i to location j, 0 otherwise. Type: GRB.BINARY.
    -   (For subtour elimination, auxiliary variables such as `u[i]` may be introduced for locations A, B, C; these represent the order in which each customer is visited. Type: GRB.INTEGER.)
5.  **Identify Parameters (from Schema):**
    -   Distance between locations i and j will come from the corresponding entry in DistanceMatrix.csv, using only the rows and columns for Depot, A, B, and C.
6.  **Formulate Objective:** Minimize the total travel distance, i.e., minimize the sum over all pairs (i, j) of distance[i, j] * x[i, j], where the sum is over all ordered pairs of the four relevant locations.
7.  **Formulate Constraints:**
    -   Constraint 1 (Visit Each Location Once): For each location (A, B, C), the van must arrive exactly once and depart exactly once (i.e., sum over j of x[i, j] = 1 for each i, and sum over i of x[i, j] = 1 for each j, excluding self-loops).
    -   Constraint 2 (Depot Start and End): The van must depart from the depot exactly once and return to the depot exactly once.
    -   Constraint 3 (Subtour Elimination): To prevent disconnected cycles (subtours), add constraints (e.g., Miller-Tucker-Zemlin constraints) using auxiliary variables `u[i]` for customers A, B, C.
    -   (No time windows or service times are needed, as per the query.)
[Abstract Model Plan END]