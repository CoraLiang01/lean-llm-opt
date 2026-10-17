[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine the optimal activation, startup, and transported-weight schedule for 10 candidate trucks over 4 periods to meet customer demand (with a 10% spare-capacity buffer), minimize total startup and transportation costs, and respect operational constraints such as minimum up/down times, ramping limits, and truck capacities.
2.  **Identify Model Type:** Based on the query, this is a Mixed-Integer Programming (MIP) problem with fixed-charge (startup) and ramping constraints.
3.  **Define Index Sets:** The primary indices are:
    - Trucks: \( i \in \{1, ..., 10\} \) (from 'truck_id')
    - Periods: \( t \in \{1, 2, 3, 4\} \)
4.  **Define Decision Variables:**
    - \( x_{i,t} \) = Amount of goods (kg) transported by truck \( i \) in period \( t \). Type: GRB.CONTINUOUS, \( x_{i,t} \geq 0 \).
    - \( y_{i,t} \) = 1 if truck \( i \) is active (on) in period \( t \), 0 otherwise. Type: GRB.BINARY.
    - \( s_{i,t} \) = 1 if truck \( i \) is started up at the beginning of period \( t \), 0 otherwise. Type: GRB.BINARY.
5.  **Identify Parameters (from Schema):**
    - Truck maximum capacity: \( Q_i \) (from 'Q')
    - Startup cost: \( S_i \) (from 'S')
    - Unit transportation cost: \( C_i \) (from 'C')
    - Customer demand per period: \( d_t \) (from 'd1', 'd2', 'd3', 'd4'; same for all trucks)
6.  **Formulate Objective:** Minimize total cost, which is the sum over all trucks and periods of:
    - Startup costs: \( S_i \cdot s_{i,t} \) (incurred when a truck is started in period \( t \))
    - Transportation costs: \( C_i \cdot x_{i,t} \)
    - Objective: Minimize \( \sum_{i} \sum_{t} (S_i \cdot s_{i,t} + C_i \cdot x_{i,t}) \)
7.  **Formulate Constraints:**
    - **Demand Satisfaction:** For each period \( t \), \( \sum_{i} x_{i,t} \geq d_t \).
    - **Spare-Capacity Buffer:** For each period \( t \), \( \sum_{i} x_{i,t} \leq 0.9 \cdot \sum_{i} Q_i \cdot y_{i,t} \).
    - **Truck Capacity:** For all \( i, t \), \( x_{i,t} \leq Q_i \cdot y_{i,t} \).
    - **Activation Consistency:** For all \( i, t \), \( x_{i,t} = 0 \) if \( y_{i,t} = 0 \).
    - **Startup Definition:** For all \( i, t \), \( s_{i,t} = y_{i,t} - y_{i,t-1} \) (with \( y_{i,0} = 0 \) since all trucks are initially off).
    - **Minimum Up-Time:** If \( s_{i,t} = 1 \), then \( y_{i,t+1} = 1 \) (trucks started in period 4 are not allowed).
    - **No Startup in Last Period:** For all \( i \), \( s_{i,4} = 0 \).
    - **Minimum Down-Time:** If \( y_{i,t-1} = 1 \) and \( y_{i,t} = 0 \), then \( y_{i,t+1} = 0 \) (truck must stay off for at least two periods after shutdown; cannot restart before \( t+2 \)).
    - **Ramping Constraint:** For all \( i, t = 2,3,4 \), \( |x_{i,t} - x_{i,t-1}| \leq 300 \).
    - **Initial State:** For all \( i \), \( y_{i,0} = 0 \) (all trucks off before period 1).
    - **Non-Negativity:** For all \( i, t \), \( x_{i,t} \geq 0 \).
[Abstract Model Plan END]