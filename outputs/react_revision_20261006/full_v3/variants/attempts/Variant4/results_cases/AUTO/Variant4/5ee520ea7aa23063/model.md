[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of emergency service centers to open, so that every required district is covered by at least one open center, while minimizing the total opening cost. Each center can cover a specific set of districts, and each district must be covered.
2.  **Identify Model Type:** Based on the query, this is a Binary Set Covering (Mixed Integer Programming, MIP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Centers (from service_centers.csv, column 'Center')
    - Districts (from districts.csv, column 'District')
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if center i is opened, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: 'OpeningCost' column in service_centers.csv (cost to open each center).
    -   Coverage mapping: 'CoveredDistricts' column in service_centers.csv (semicolon-separated list of districts each center can cover).
    -   Required coverage set: 'District' column in districts.csv (list of all districts that must be covered).
6.  **Formulate Objective:** Minimize the total opening cost, i.e., minimize the sum over all centers of (OpeningCost[i] * y[i]).
7.  **Formulate Constraints:**
    -   Coverage Constraint: For each district j in the required set, the sum over all centers i that can cover district j of y[i] must be at least 1. (Every district must be covered by at least one open center.)
    -   Binary Restriction: For each center i, y[i] ∈ {0,1}.
[Abstract Model Plan END]