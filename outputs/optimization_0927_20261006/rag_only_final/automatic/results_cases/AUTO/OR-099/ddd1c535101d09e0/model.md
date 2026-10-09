Mathematical Model:

Sets:
- I = {1,2,...,11} (warehouses)
- J = {1,2,...,11} (stores)

Parameters:
- fi: Opening cost for warehouse i
  f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]
- si: Capacity of warehouse i
  s = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]
- dj: Demand of store j
  d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]
- ci,j: Transportation cost per unit from warehouse i to store j
  C = [
    [12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15],
    [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16],
    [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17],
    [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18],
    [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17],
    [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19],
    [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14],
    [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18],
    [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18],
    [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19],
    [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]
  ]

Decision Variables:
- yi ∈ {0,1}, for i ∈ I: 1 if warehouse i is opened, 0 otherwise
- xi,j ≥ 0, for i ∈ I, j ∈ J: units supplied from warehouse i to store j

Objective:
Minimize total cost:
minimize
  ∑_{i=1}^{11} fi * yi + ∑_{i=1}^{11} ∑_{j=1}^{11} ci,j * xi,j

Subject to:
1. Demand satisfaction:
   For each j = 1,...,11:
     ∑_{i=1}^{11} xi,j = dj

2. Warehouse capacity:
   For each i = 1,...,11:
     ∑_{j=1}^{11} xi,j ≤ si * yi

3. Variable domains:
   yi ∈ {0,1} for all i
   xi,j ≥ 0 for all i, j

All parameters (vectors and matrices) are as listed above.