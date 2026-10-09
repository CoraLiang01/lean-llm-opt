[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect truck-specific capacity, startup, ramping, and minimum up/down time constraints.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping/minimum up/down time constraints.
3.  **Define Index Sets:** The primary indices are Trucks (i ∈ {1,…,10}) and Periods (t ∈ {1,2,3,4}).
4.  **Define Decision Variables:**
    -   `x[i,t]` = Amount of goods (kg) transported by truck i in period t. Type: GRB.CONTINUOUS, x[i,t] ≥ 0.
    -   `y[i,t]` = 1 if truck i is active (on) in period t, 0 otherwise. Type: GRB.BINARY.
    -   `z[i,t]` = 1 if truck i is started up at the beginning of period t, 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    -   Truck maximum capacity: `Q[i]` (from column 'Q').
    -   Truck startup cost: `S[i]` (from column 'S').
    -   Truck unit transportation cost: `C[i]` (from column 'C').
    -   Customer demand per period: `d[t]` (from columns 'd1', 'd2', 'd3', 'd4'; same for all trucks).
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of (startup cost × startup indicator) plus (unit transportation cost × transported amount):  
    Minimize ∑_{i,t} S[i] * z[i,t] + ∑_{i,t} C[i] * x[i,t].
7.  **Formulate Constraints:**
    -   **Demand Satisfaction:** For each period t, the total transported weight across all trucks must be at least the customer demand:  
        ∑_{i} x[i,t] ≥ d[t].
    -   **Spare-Capacity Buffer:** For each period t, the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:  
        ∑_{i} x[i,t] ≤ 0.9 * ∑_{i} Q[i] * y[i,t].
    -   **Truck Capacity:** For each truck i and period t, the transported weight cannot exceed the truck's capacity if active, and must be zero if inactive:  
        0 ≤ x[i,t] ≤ Q[i] * y[i,t].
    -   **Startup Definition:** For each truck i and period t, startup occurs if the truck is off in t-1 and on in t:  
        z[i,1] = y[i,1] (since all trucks are initially off);  
        For t > 1: z[i,t] ≥ y[i,t] - y[i,t-1].
    -   **Minimum Up-Time:** If a truck is started in period t, it must remain active in period t and t+1 (cannot be started in period 4):  
        For t ∈ {1,2,3}: z[i,t] ≤ y[i,t] and z[i,t] ≤ y[i,t+1];  
        z[i,4] = 0 (no startups allowed in period 4).
    -   **Minimum Down-Time:** If a truck is shut down in period t (i.e., y[i,t-1]=1, y[i,t]=0), it must remain off in t and t+1, and cannot be restarted before t+2:  
        For t ∈ {1,2,3}: y[i,t-1] - y[i,t] ≤ 1 - y[i,t+1];  
        For t ∈ {1,2}: y[i,t-1] - y[i,t] ≤ 1 - y[i,t+2].
    -   **Ramping Constraint:** For each truck i and periods t=1 to 3, the change in transported weight between adjacent periods cannot exceed 300 kg (including transitions to/from zero):  
        |x[i,t+1] - x[i,t]| ≤ 300.
    -   **Initial State:** All trucks are off before period 1 (y[i,0]=0 for all i).
[Abstract Model Plan END]