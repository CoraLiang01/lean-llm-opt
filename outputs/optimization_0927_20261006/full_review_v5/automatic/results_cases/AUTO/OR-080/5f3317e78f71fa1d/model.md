[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand, minimize total startup and transportation costs, and satisfy operational constraints including minimum up/down times, ramping, capacity, and buffer requirements.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are Trucks (i ∈ {1,…,10}) and Periods (t ∈ {1,2,3,4}).
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck i in period t. Type: GRB.CONTINUOUS, x[i,t] ≥ 0.
    -   `y[i,t]` = 1 if truck i is active (on) in period t, 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck i is started up at the beginning of period t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q[i]` (from 'Q' column).
    -   Truck startup cost: `S[i]` (from 'S' column).
    -   Truck unit transportation cost: `C[i]` (from 'C' column).
    -   Customer demand per period: `d[t]` (from 'd1', 'd2', 'd3', 'd4' columns).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of (startup cost × startup indicator) plus (unit transportation cost × transported amount):  
    Minimize ∑₍i,t₎ [S[i] * z[i,t] + C[i] * x[i,t]].
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period t, the total transported weight across all trucks must be at least the customer demand:  
        ∑₍i₎ x[i,t] ≥ d[t].
    -   **Spare-Capacity Buffer:** For each period t, the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:  
        ∑₍i₎ x[i,t] ≤ 0.9 * ∑₍i₎ Q[i] * y[i,t].
    -   **Truck Capacity:** For each truck i and period t, transported weight cannot exceed truck capacity when active, and must be zero when inactive:  
        0 ≤ x[i,t] ≤ Q[i] * y[i,t].
    -   **Startup Definition:** For each truck i and period t, startup occurs if the truck is off in t-1 and on in t:  
        z[i,1] = y[i,1] (since all trucks are initially off);  
        For t > 1: z[i,t] ≥ y[i,t] - y[i,t-1]; z[i,t] ≥ 0.
    -   **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods; trucks cannot be started in period 4:  
        For all i: z[i,4] = 0;  
        For all i, t ∈ {1,2,3}: y[i,t+1] ≥ y[i,t] - z[i,t] (enforces at least two periods on after startup).
    -   **Minimum Down-Time:** If a truck is shut down in period t (i.e., y[i,t-1]=1, y[i,t]=0), it must remain off in t and t+1, and cannot be restarted before t+2:  
        For all i, t ∈ {1,2,3}: y[i,t] + y[i,t+1] ≤ 1 if y[i,t-1]=1 and y[i,t]=0.
    -   **Ramping Constraint:** For each truck i and periods t=2,3,4, the change in transported weight between adjacent periods cannot exceed 300 kg (including transitions to/from zero):  
        |x[i,t] - x[i,t-1]| ≤ 300.
    -   **Initial State:** All trucks are off before period 1 (y[i,0]=0), and transported weight is zero before period 1 (x[i,0]=0).
    -   **Non-Negativity and Binary:** All x[i,t] ≥ 0; all y[i,t], z[i,t] ∈ {0,1}.
[Abstract Model Plan END]