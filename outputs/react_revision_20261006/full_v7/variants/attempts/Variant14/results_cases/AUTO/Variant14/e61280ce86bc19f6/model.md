[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a minimum-cost set of emergency repair depots such that every service zone is covered by at least one opened depot. Each depot has an opening cost and covers a specified subset of service zones. The goal is to minimize the total opening cost while ensuring all service zones are covered.
2.  **Identify Model Type:** Based on the query, this is a Set Covering Problem, formulated as a Mixed-Integer Programming (MIP) model.
3.  **Define Index Sets:** The primary indices are:
    - Depots (from facility_sites.csv, column 'Center')
    - Service Zones (from service_zones.csv, column 'Zone')
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if depot i is opened, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Depot opening costs from facility_sites.csv, column 'OpeningCost'.
    -   Coverage mapping: For each depot (facility_sites.csv, 'CoveredDistricts'), the set of service zones it covers (parsed from the semicolon-separated list).
    -   Constraint RHS: Each service zone (from service_zones.csv, column 'Zone') must be covered at least once (RHS = 1).
6.  **Formulate Objective:** Minimize the total opening cost, i.e., sum over all depots of (OpeningCost[i] * y[i]).
7.  **Formulate Constraints:**
    -   Coverage Constraint: For each service zone j, the sum over all depots i that cover zone j of y[i] must be at least 1. (This ensures every service zone is covered by at least one opened depot.)
    -   Binary Restriction: For each depot i, y[i] ∈ {0, 1}.
[Abstract Model Plan END]