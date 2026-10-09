[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect truck-specific capacity, startup, ramping, and minimum up/down time constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping/minimum up/down time constraints.
3.  **Define Index Sets:** The primary indices are Trucks (i ∈ {1,…,10}) and Periods (t ∈ {1,2,3,4}).
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck i in period t. Type: GRB.CONTINUOUS, x[i,t] ≥ 0.
    -   `y[i,t]` = 1 if truck i is active (on) in period t, 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck i is started up at the beginning of period t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: Q[i] (from 'Q' column).
    -   Truck startup cost: S[i] (from 'S' column).
    -   Truck unit transportation cost: C[i] (from 'C' column).
    -   Customer demand per period: d[t] (from 'd1', 'd2', 'd3', 'd4' columns; same for all trucks).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of (startup cost S[i] × z[i,t]) plus (unit cost C[i] × x[i,t]).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each period t, sum over all trucks of x[i,t] ≥ d[t].
    -   Spare-Capacity Buffer: For each period t, sum over all trucks of x[i,t] ≤ 0.9 × sum over i of (Q[i] × y[i,t]).
    -   Truck Capacity: For all i, t, x[i,t] ≤ Q[i] × y[i,t]; x[i,t] = 0 if y[i,t] = 0.
    -   Startup Logic: For all i, t, z[i,t] = y[i,t] - y[i,t-1] (with y[i,0] = 0 since all trucks are initially off).
    -   Startup Restriction: For all i, z[i,4] = 0 (no truck may be started in period 4).
    -   Minimum Up-Time: For all i, if z[i,t] = 1 (truck started at t), then y[i,t+1] = 1 (must stay on at least two consecutive periods; for t=3, only t+1=4 is relevant).
    -   Minimum Down-Time: For all i, if truck is shut down at t (i.e., y[i,t-1]=1, y[i,t]=0), then y[i,t+1]=0 and y[i,t+2]=0 (cannot restart before t+2); for t=3, only t+1=4 is relevant.
    -   Ramping Constraint: For all i, t=2..4, |x[i,t] - x[i,t-1]| ≤ 300 kg (including transitions to/from zero).
    -   Inactivity Constraint: For all i, t, if y[i,t]=0 then x[i,t]=0.
[Abstract Model Plan END]