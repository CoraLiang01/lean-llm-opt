[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand, minimize total startup and transportation costs, and satisfy operational constraints including minimum up/down times, ramping limits, and spare-capacity requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are Trucks (i ∈ {1,…,10}) and Periods (t ∈ {1,2,3,4}).
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of weight (kg) transported by truck i in period t. Type: GRB.CONTINUOUS, x[i,t] ≥ 0.
    -   `y[i,t]` = 1 if truck i is active (on) in period t, 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck i is started up at the beginning of period t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: Q[i] (from 'Q' column).
    -   Startup cost: S[i] (from 'S' column).
    -   Unit transportation cost: C[i] (from 'C' column).
    -   Customer demand per period: d[t] (from 'd1', 'd2', 'd3', 'd4' columns).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of startup costs (S[i] * z[i,t]) plus transportation costs (C[i] * x[i,t]).
7.  **Formulate Constraints:**
    -   Demand Satisfaction: For each period t, the sum over all trucks of x[i,t] ≥ d[t].
    -   Spare-Capacity Buffer: For each period t, sum over i of x[i,t] ≤ 0.9 * sum over i of Q[i] * y[i,t].
    -   Truck Capacity: For all i, t, x[i,t] ≤ Q[i] * y[i,t].
    -   Inactive Truck: For all i, t, x[i,t] = 0 if y[i,t] = 0.
    -   Startup Definition: For all i, t ≥ 1, z[i,t] = y[i,t] - y[i,t-1] (with y[i,0] = 0 since all trucks are initially off).
    -   Minimum Up-Time: If truck i is started in period t (z[i,t]=1), then y[i,t+1] = 1 (for t=1,2,3), i.e., once started, must stay on at least two consecutive periods; trucks cannot be started in period 4.
    -   Minimum Down-Time: If truck i is shut down in period t (y[i,t-1]=1, y[i,t]=0), then y[i,t+1]=0 and y[i,t+2]=0 (for t=1,2), i.e., after shutdown, must stay off for at least two periods and cannot restart before t+2.
    -   Ramping Constraint: For all i, t=2,3,4, |x[i,t] - x[i,t-1]| ≤ 300 kg.
    -   Initial State: For all i, y[i,0]=0 (all trucks off before period 1).
    -   No Startup in Last Period: For all i, z[i,4]=0 (cannot start in period 4).
[Abstract Model Plan END]