[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 consecutive periods to minimize total startup and transportation costs, while satisfying customer demand, truck operational constraints (minimum up/down times, ramping, capacity, and activation logic), and a spare-capacity buffer.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, \ldots, 10\} \) (from 'truck_id' in parameters.csv)
    - Periods: \( t \in \{1, 2, 3, 4\} \) (time periods)
4.  **Define Decision Variables:**
    - \( y_{i,t} \): Binary variable; 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( s_{i,t} \): Binary variable; 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( x_{i,t} \): Continuous variable; weight (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS.
5.  **Identify Parameters (from Schema):**
    - Maximum capacity per truck: \( Q_i \) (column 'Q')
    - Startup cost per truck: \( S_i \) (column 'S')
    - Unit transportation cost per truck: \( C_i \) (column 'C')
    - Customer demand per period: \( d_t \) (columns 'd1', 'd2', 'd3', 'd4'; same for all trucks)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( S_i \cdot s_{i,t} \) (incurred when a truck is started in period \( t \))
    - Transportation costs: \( C_i \cdot x_{i,t} \) (incurred for each kg transported)
    - Objective: Minimize \( \sum_{i} \sum_{t} (S_i \cdot s_{i,t} + C_i \cdot x_{i,t}) \)
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), the total transported weight must meet or exceed customer demand: \( \sum_{i} x_{i,t} \geq d_t \).
    - **Spare-Capacity Buffer:** For each period \( t \), the total transported weight cannot exceed 90% of the combined maximum capacity of active trucks: \( \sum_{i} x_{i,t} \leq 0.9 \cdot \sum_{i} Q_i \cdot y_{i,t} \).
    - **Truck Capacity:** For each truck \( i \) and period \( t \), transported weight cannot exceed truck capacity when active, and must be zero when inactive: \( 0 \leq x_{i,t} \leq Q_i \cdot y_{i,t} \).
    - **Startup Logic:** For each truck \( i \) and period \( t \), startup occurs if the truck is off in \( t-1 \) and on in \( t \): \( s_{i,t} \geq y_{i,t} - y_{i,t-1} \), with \( y_{i,0} = 0 \) (all trucks initially off).
    - **Minimum Up-Time:** If a truck is started in period \( t \), it must remain active in period \( t \) and \( t+1 \): \( y_{i,t+1} \geq s_{i,t} \) for \( t = 1,2,3 \); trucks cannot be started in period 4 (\( s_{i,4} = 0 \)).
    - **Minimum Down-Time:** If a truck is shut down in period \( t \) (i.e., \( y_{i,t-1}=1, y_{i,t}=0 \)), it must remain off in periods \( t \) and \( t+1 \), and cannot be restarted before \( t+2 \): \( y_{i,t} + y_{i,t+1} \leq 1 - (y_{i,t-1} - y_{i,t}) \) for \( t = 1,2,3 \).
    - **Ramping Constraint:** For each truck \( i \) and periods \( t=2,3,4 \), the change in transported weight between adjacent periods cannot exceed 300 kg: \( |x_{i,t} - x_{i,t-1}| \leq 300 \).
    - **Initial State:** All trucks are off before period 1: \( y_{i,0} = 0 \), \( x_{i,0} = 0 \).
    - **No Startup in Last Period:** \( s_{i,4} = 0 \) for all \( i \).
    - **Inactivity Implies Zero Transport:** If \( y_{i,t} = 0 \), then \( x_{i,t} = 0 \).
[Abstract Model Plan END]