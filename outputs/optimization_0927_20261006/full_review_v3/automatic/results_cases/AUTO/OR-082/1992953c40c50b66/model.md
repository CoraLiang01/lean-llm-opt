[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal sequence for a delivery van to visit three customer locations (A, B, C) exactly once each, starting and ending at the depot, so as to minimize the total travel distance. The pairwise distances between all locations (including the depot) are provided in DistanceMatrix.csv.
2.  **Identify Model Type:** Based on the query, this is a Traveling Salesman Problem (TSP), which is a Mixed Integer Programming (MIP) problem.
3.  **Define Index Sets:** The primary indices are the set of locations to be visited: {Depot, A, B, C}.
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if the route travels directly from location i to location j, 0 otherwise. Type: GRB.BINARY.
    -   (If needed for subtour elimination) `u[i]` = position of location i in the tour (for i ≠ Depot). Type: GRB.INTEGER.
5.  **Identify Parameters (from Schema):**
    -   Distance between locations i and j will come from the corresponding entry in DistanceMatrix.csv, using the intersection of row 'Unnamed: 0' = i and column j.
6.  **Formulate Objective:** Minimize the total travel distance, i.e., minimize the sum over all pairs (i, j) of Distance[i, j] * x[i, j], where i ≠ j and both i, j ∈ {Depot, A, B, C}.
7.  **Formulate Constraints:**
    -   Constraint 1 (Visit Each Location Once): For each location (including the depot), the sum of x[i, j] over all j ≠ i equals 1 (each location is departed from exactly once), and the sum over all i ≠ j of x[i, j] equals 1 (each location is arrived at exactly once).
    -   Constraint 2 (Subtour Elimination): For all i, j ∈ {A, B, C}, i ≠ j, enforce u[i] - u[j] + 3 * x[i, j] ≤ 2 to prevent subtours (if using MTZ formulation).
    -   Constraint 3 (Variable Domains): x[i, j] ∈ {0,1} for all i ≠ j; u[i] ∈ {1, 2, 3} for i ∈ {A, B, C}.
[Abstract Model Plan END]