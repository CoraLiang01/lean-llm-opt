[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to select a minimum-cost set of emergency repair depots so that every service zone is covered by at least one opened depot. Each depot has an opening cost and covers a specific subset of service zones. The goal is to minimize the total opening cost while ensuring all zones are covered.
2.  **Identify Model Type:** Based on the query, this is a Set Covering Problem, formulated as a Mixed-Integer Programming (MIP) model with binary variables.
3.  **Define Index Sets:** The primary indices are:
    - Depots (from 'Center' in facility_sites.csv)
    - Service Zones (from 'Zone' in service_zones.csv)
4.  **Define Decision Variables:**
    -   `y[i]` = 1 if depot i is opened, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Objective coefficients: Depot opening costs from 'OpeningCost' column in facility_sites.csv.
    -   Coverage mapping: For each depot (row in facility_sites.csv), the set of service zones it covers from 'CoveredDistricts' (semicolon-separated list).
    -   Constraint RHS: Each service zone (from service_zones.csv) must be covered at least once (RHS = 1).
6.  **Formulate Objective:** Minimize the total opening cost, i.e., sum over all depots of (OpeningCost[i] * y[i]).
7.  **Formulate Constraints:**
    -   Coverage Constraint: For each service zone j (from service_zones.csv), the sum of y[i] over all depots i that cover zone j (i.e., where zone j is in CoveredDistricts[i]) must be at least 1. This ensures every zone is covered by at least one opened depot.
    -   Binary Restriction: For all depots i, y[i] ∈ {0,1}.
[Abstract Model Plan END]