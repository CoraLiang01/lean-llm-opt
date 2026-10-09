Let:
- S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12} be the set of distribution centers.
- C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12} be the set of customer groups.

Parameters:
- demand_c: daily demand for customer group c ∈ C (from customer_demand.csv)
  - demand_C1 = 52
  - demand_C2 = 80
  - demand_C3 = 392
  - demand_C4 = 103
  - demand_C5 = 32
  - demand_C6 = 1426
  - demand_C7 = 1024
  - demand_C8 = 2736
  - demand_C9 = 1129
  - demand_C10 = 676
  - demand_C11 = 2631
  - demand_C12 = 31

- supply_s: supply capacity for distribution center s ∈ S (from supply_capacity.csv)
  - supply_S1 = 58
  - supply_S2 = 32
  - supply_S3 = 6161
  - supply_S4 = 4
  - supply_S5 = 47
  - supply_S6 = 178
  - supply_S7 = 142
  - supply_S8 = 164
  - supply_S9 = 1011
  - supply_S10 = 6
  - supply_S11 = 7081
  - supply_S12 = 948

- cost_{s,c}: transportation cost per unit from distribution center s to customer group c (from transportation_costs.csv):

| s  | C1         | C2         | C3         | C4         | C5         | C6         | C7         | C8         | C9         | C10        | C11        | C12        |
|----|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|S1  |134.7288244 |72.30045141 |37.97599274 |611.8465650 |1650.335390 |32.97044487 |34.99837373 |73.25207370 |165.7520898 |82.52035828 |1538.830198 |2320.114648 |
|S2  |23.83536919 |1128.633287 |187.0545959 |1.607733713 |2227.703699 |72.44693149 |12.60074797 |1078.772955 |383.7220569 |82.81152504 |1702.172476 |1925.844697 |
|S3  |1138.498760 |231.8983422 |44.15682267 |962.5481641 |980.1306032 |1107.515751 |741.7073424 |136.7020440 |1182.516112 |664.1846793 |36.70238978 |1405.050692 |
|S4  |1043.756280 |24.20701606 |1120.989044 |1027.543764 |893.4457578 |1244.262678 |980.0297708 |513.5650272 |977.6321548 |642.5520451 |437.9556823 |76.12498669 |
|S5  |4.939930662 |1278.181574 |549.7570743 |21.33547411 |98.32689280 |452.0903463 |595.5981934 |70.84029747 |0.001236242 |1783.086568 |1619.615466 |116.0849791 |
|S6  |2105.539860 |1340.977056 |2077.274728 |2202.251540 |20.53369588 |2514.098360 |2393.213467 |1197.765543 |102.3244025 |788.4640306 |818.9532260 |17.24964579 |
|S7  |61.36183063 |113.2371696 |50.23107648 |1219.195108 |869.8689864 |58.37177230 |957.3213131 |168.9964798 |1363.342714 |519.3858364 |483.3345987 |1412.765288 |
|S8  |1169.364099 |1037.217074 |732.3577707 |865.2255769 |1510.029619 |780.8537898 |860.7971696 |935.9126433 |61.82504910 |72.78392607 |1479.113720 |65.87463475 |
|S9  |7.936521667 |1357.538761 |628.3001825 |25.24514182 |1760.008525 |604.6583536 |696.6774833 |1586.409318 |89.68006943 |1580.726256 |1423.561541 |2000.563800 |
|S10 |1685.340976 |437.3938491 |1568.893994 |1486.926897 |498.3754447 |1493.386021 |70.17952879 |526.9999923 |1527.660660 |2.676413574 |202.7267124 |45.95308313 |
|S11 |937.1094078 |903.0141416 |264.7208999 |21.82334018 |1661.105486 |18.59349094 |372.0817876 |956.6955599 |42.91201853 |1274.322753 |1574.796150 |1826.428257 |
|S12 |1685.757685 |377.3256673 |1347.016273 |1737.020031 |23.61217798 |83.08521829 |1476.125824 |530.0021927 |1782.844263 |0.010486218 |11.16932045 |963.9941815 |

Decision variables:
- x_{s,c} ≥ 0: quantity of goods shipped from distribution center s ∈ S to customer group c ∈ C.

Objective:
Minimize total transportation cost:
\[
\min \sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
\]
That is,
\[
\min \Bigg(
\sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
\Bigg)
\]
where all cost_{s,c} are as given above.

Subject to:

1. Demand satisfaction for each customer group:
\[
\sum_{s \in S} x_{s,c} = demand_c \quad \forall c \in C
\]
That is, for each c ∈ {C1,...,C12}:
- \(\sum_{s \in S} x_{s,C1} = 52\)
- \(\sum_{s \in S} x_{s,C2} = 80\)
- ...
- \(\sum_{s \in S} x_{s,C12} = 31\)

2. Supply capacity for each distribution center:
\[
\sum_{c \in C} x_{s,c} \leq supply_s \quad \forall s \in S
\]
That is, for each s ∈ {S1,...,S12}:
- \(\sum_{c \in C} x_{S1,c} \leq 58\)
- \(\sum_{c \in C} x_{S2,c} \leq 32\)
- ...
- \(\sum_{c \in C} x_{S12,c} \leq 948\)

3. Non-negativity:
\[
x_{s,c} \geq 0 \quad \forall s \in S, c \in C
\]

Summary:
Minimize
\[
\sum_{s \in S} \sum_{c \in C} cost_{s,c} \cdot x_{s,c}
\]
subject to
\[
\sum_{s \in S} x_{s,c} = demand_c \quad \forall c \in C
\]
\[
\sum_{c \in C} x_{s,c} \leq supply_s \quad \forall s \in S
\]
\[
x_{s,c} \geq 0 \quad \forall s \in S, c \in C
\]
with all identifiers and coefficients as listed above, in source order.