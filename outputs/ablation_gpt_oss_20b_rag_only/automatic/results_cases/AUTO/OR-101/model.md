**Mathematical Model**

```
SETS
    D   :  {A,B,I,J}          ! devices
    P   :  {P1,P2,…,P111}     ! products

PARAMETERS
    profit[p]   >= 0          ! unit profit of product p
    time[d,p]   >= 0          ! processing time of product p on device d
    capacity[d] >= 0          ! monthly capacity of device d

VARIABLES
    x[p]        >= 0          ! production quantity of product p

OBJ:  maximize  Z = Σ_{p∈P} profit[p] * x[p]

CONSTRAINTS
    cap[d] :  Σ_{p∈P} time[d,p] * x[p]  ≤  capacity[d]      ∀ d∈D
```

**Data Mapping (CSVQA_DATA)**  

| Parameter | Table ID          | Key Column | Value Column(s) |
|-----------|-------------------|------------|-----------------|
| profit[p] | file_2_view_0     | Product    | Unit_Profit     |
| time[d,p] | file_0_view_0     | Device     | P1 … P111       |
| capacity[d] | file_1_view_0   | Device     | Monthly_Capacity|

All parameters are continuous; decision variables `x[p]` are continuous non‑negative. No additional constraints are required.