[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over four consecutive periods to meet period-specific customer demands (with a 10% spare-capacity buffer), while minimizing total startup and transportation costs. The solution must respect truck capacity, startup costs, minimum up/down time, ramping (change in load), and activation rules.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \text{Trucks} \) (from 'truck_id' in parameters.csv, 10 trucks)
    - Periods: \( t \in \{1,2,3,4\} \)
4.  **Define Decision Variables:**
    - \( y_{i,t} \) = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( u_{i,t} \) = 1 if truck \( i \) is started up (turned on from off) at the start of period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( z_{i,t} \) = 1 if truck \( i \) is shut down (turned off from on) at the start of period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( x_{i,t} \) = weight transported by truck \( i \) in period \( t \) (kg). Type: GRB.CONTINUOUS, \( \geq 0 \).
5.  **Identify Parameters (from Schema):**
    - Maximum capacity per truck: \( Q_i \) (from 'Q')
    - Startup cost per truck: \( S_i \) (from 'S')
    - Unit transportation cost per truck: \( C_i \) (from 'C')
    - Period demands: \( d_t \) (from 'd1', 'd2', 'd3', 'd4'; same for all trucks)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( S_i \cdot u_{i,t} \)
    - Transportation costs: \( C_i \cdot x_{i,t} \)
    Thus, Objective = \( \sum_{i,t} S_i \cdot u_{i,t} + \sum_{i,t} C_i \cdot x_{i,t} \)
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), \( \sum_{i} x_{i,t} \geq d_t \)
    - **Spare-Capacity Buffer:** For each period \( t \), \( \sum_{i} x_{i,t} \leq 0.9 \cdot \sum_{i} Q_i \cdot y_{i,t} \)
    - **Truck Capacity:** For all \( i, t \), \( 0 \leq x_{i,t} \leq Q_i \cdot y_{i,t} \)
    - **Startup Logic:** For all \( i, t \), \( u_{i,t} \geq y_{i,t} - y_{i,t-1} \) (with \( y_{i,0} = 0 \) since all trucks are initially off)
    - **No Startup in Last Period:** For all \( i \), \( u_{i,4} = 0 \)
    - **Minimum Up-Time:** If a truck is started in period \( t \), it must remain on in period \( t \) and \( t+1 \): \( y_{i,t+1} \geq u_{i,t} \) for \( t = 1,2,3 \)
    - **Shutdown Logic:** For all \( i, t \), \( z_{i,t} \geq y_{i,t-1} - y_{i,t} \) (with \( y_{i,0} = 0 \))
    - **Minimum Down-Time:** If a truck is shut down in period \( t \), it must remain off in periods \( t \) and \( t+1 \): \( y_{i,t+1} \leq 1 - z_{i,t} \), \( y_{i,t+2} \leq 1 - z_{i,t} \) for applicable \( t \)
    - **No Startup Before Allowed:** After shutdown at \( t \), \( u_{i,t+1} = 0 \)
    - **Ramping Constraint:** For all \( i, t=2,3,4 \), \( |x_{i,t} - x_{i,t-1}| \leq 300 \)
    - **Zero Load When Off:** For all \( i, t \), \( x_{i,t} = 0 \) if \( y_{i,t} = 0 \)
[Abstract Model Plan END]