[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a subset of emergency service centers to open, such that every required district is covered by at least one opened center, while minimizing the total opening cost. Each center can cover a specific set of districts, and each district must be covered.
2.  **Identify Model Type:** Based on the query, this is a Binary Set Covering (Mixed Integer Programming, MIP) problem.
3.  **Define Index Sets:** The primary indices are:
    - Centers (from service_centers.csv, field 'Center')
    - Districts (from districts.csv, field 'District')
4.  **Define Decision Variables:**
    - `y[i]` = 1 if center i is opened, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Objective coefficients: 'OpeningCost' column from service_centers.csv, indexed by center.
    - Coverage mapping: 'CoveredDistricts' column from service_centers.csv, which lists (semicolon-separated) the districts each center can cover.
    - Required coverage set: 'District' column from districts.csv, listing all districts that must be covered.
6.  **Formulate Objective:** Minimize the total opening cost, i.e., minimize the sum over all centers of (OpeningCost for center i) × y[i].
7.  **Formulate Constraints:**
    - Coverage Constraint: For each district j in the required set, the sum over all centers i that can cover district j of y[i] must be at least 1 (i.e., every district is covered by at least one opened center).
    - Binary Restriction: For each center i, y[i] ∈ {0,1}.
[Abstract Model Plan END]