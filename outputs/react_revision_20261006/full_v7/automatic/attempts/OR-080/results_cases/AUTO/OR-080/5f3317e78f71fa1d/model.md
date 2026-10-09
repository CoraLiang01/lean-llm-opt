[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect operational constraints such as minimum up/down times, ramping limits, and truck capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, 2, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    - \( x[i, t] \): Amount of weight (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS, \( x[i, t] \geq 0 \).
    - \( y[i, t] \): Binary variable; 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( s[i, t] \): Binary variable; 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Truck maximum capacity: \( Q[i] \) (from column 'Q')
    - Startup cost: \( S[i] \) (from column 'S')
    - Unit transportation cost: \( C[i] \) (from column 'C')
    - Customer demand per period: \( d[t] \) (from columns 'd1', 'd2', 'd3', 'd4'; same for all trucks)
6.  **Formulate Objective:** Minimize total cost, which is the sum of all startup costs (for each truck and period when started) plus the sum of all transportation costs (for each truck and period):
    - \( \text{Minimize} \sum_{i=1}^{10} \sum_{t=1}^{4} S[i] \cdot s[i, t] + C[i] \cdot x[i, t] \)
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), the total transported weight must meet or exceed customer demand:
        - \( \sum_{i=1}^{10} x[i, t] \geq d[t] \)
    - **Spare-Capacity Buffer:** For each period \( t \), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks:
        - \( \sum_{i=1}^{10} x[i, t] \leq 0.9 \cdot \sum_{i=1}^{10} Q[i] \cdot y[i, t] \)
    - **Truck Capacity:** For each truck \( i \) and period \( t \), transported weight cannot exceed truck capacity if active, and must be zero if inactive:
        - \( x[i, t] \leq Q[i] \cdot y[i, t] \)
        - \( x[i, t] \geq 0 \)
    - **Startup Definition:** For each truck \( i \) and period \( t \), startup occurs if the truck is on in \( t \) but was off in \( t-1 \) (with all trucks off before period 1):
        - For \( t = 1 \): \( s[i, 1] = y[i, 1] \) (since all trucks are initially off)
        - For \( t > 1 \): \( s[i, t] \geq y[i, t] - y[i, t-1] \), \( s[i, t] \geq 0 \)
    - **Minimum Up-Time:** Once a truck is started, it must remain active for at least two consecutive periods. For each truck \( i \) and period \( t \) (where \( t \leq 3 \)), if started in \( t \), must be on in \( t+1 \):
        - \( y[i, t+1] \geq s[i, t] \) for \( t = 1, 2, 3 \)
        - No startups allowed in period 4: \( s[i, 4] = 0 \)
    - **Minimum Down-Time:** If a truck is shut down in period \( t \) (i.e., \( y[i, t-1]=1, y[i, t]=0 \)), it must remain off in periods \( t \) and \( t+1 \) (cannot be restarted before \( t+2 \)). For \( t = 2, 3 \):
        - If \( y[i, t-1]=1, y[i, t]=0 \), then \( y[i, t+1]=0 \)
        - This can be enforced by: \( y[i, t-1] - y[i, t] \leq 1 - y[i, t+1] \) for \( t = 2, 3 \)
    - **No Startup in Last Period:** \( s[i, 4] = 0 \) for all \( i \)
    - **Ramping (Change in Load):** For each truck \( i \) and periods \( t = 2, 3, 4 \), the change in transported weight between adjacent periods cannot exceed 300 kg (including transitions to/from zero):
        - \( |x[i, t] - x[i, t-1]| \leq 300 \)
    - **Inactive Truck Must Transport Zero:** For all \( i, t \), if \( y[i, t]=0 \), then \( x[i, t]=0 \) (already enforced by capacity constraint above).
[Abstract Model Plan END]