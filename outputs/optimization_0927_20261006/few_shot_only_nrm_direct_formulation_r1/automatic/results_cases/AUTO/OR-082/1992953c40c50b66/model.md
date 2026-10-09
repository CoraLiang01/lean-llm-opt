[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal sequence for a delivery van to visit three specific customer locations (A, B, C) exactly once each, starting and ending at the depot, so as to minimize the total travel distance. The pairwise distances between all locations are provided in DistanceMatrix.csv. Only the depot and locations A, B, and C are relevant; other locations in the CSV are not part of the route.
2.  **Identify Model Type:** Based on the query, this is a Traveling Salesman Problem (TSP), which is a Mixed-Integer Programming (MIP) problem.
3.  **Define Index Sets:** The primary indices are the set of locations to be visited: {Depot, A, B, C}.
4.  **Define Decision Variables:**
    -   `x[i,j]` = 1 if the route travels directly from location i to location j, 0 otherwise. Type: GRB.BINARY, for all ordered pairs (i, j) where i ≠ j and i, j ∈ {Depot, A, B, C}.
5.  **Identify Parameters (from Schema):**
    -   Distance between locations i and j: from DistanceMatrix.csv, using the intersection of row 'i' and column 'j' for i, j ∈ {Depot, A, B, C}.
6.  **Formulate Objective:** Minimize the total travel distance, i.e., minimize the sum over all pairs (i, j) of distance[i, j] * x[i, j].
7.  **Formulate Constraints:**
    -   Each location (A, B, C) is entered exactly once: For each k ∈ {A, B, C}, sum over i ≠ k of x[i, k] = 1.
    -   Each location (A, B, C) is departed exactly once: For each k ∈ {A, B, C}, sum over j ≠ k of x[k, j] = 1.
    -   The depot is departed exactly once and entered exactly once: sum over j ≠ Depot of x[Depot, j] = 1; sum over i ≠ Depot of x[i, Depot] = 1.
    -   Subtour elimination: For all proper subsets S of {A, B, C}, the number of arcs within S must be less than |S| (e.g., using Miller-Tucker-Zemlin or explicit enumeration for small sets).
[Abstract Model Plan END]