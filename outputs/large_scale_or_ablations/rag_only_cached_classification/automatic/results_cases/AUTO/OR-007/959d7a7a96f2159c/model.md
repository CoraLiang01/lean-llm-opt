Let:
- S = {S1, S2, S3, S4, S5} be the set of warehouses (supply regions)
- D = {D1, D2, D3, D4, D5} be the set of stores (demand points)
- x_{ij} = quantity shipped from warehouse i ∈ S to store j ∈ D

Parameters (from CSVs, source order preserved):

Demands:
- Demand at D1: 428
- Demand at D2: 217
- Demand at D3: 214
- Demand at D4: 380
- Demand at D5: 254

Supply capacities:
- Supply at S1: 428
- Supply at S2: 217
- Supply at S3: 214
- Supply at S4: 380
- Supply at S5: 254

Transportation costs (c_{ij}):
- From S1: c_{S1,D1} = 269.3910588020795, c_{S1,D2} = 1.4537335390933939, c_{S1,D3} = 99.60345345756605, c_{S1,D4} = 26.64078166309837, c_{S1,D5} = 9.537688956880922
- From S2: c_{S2,D1} = 9.291846876785183, c_{S2,D2} = 10.874778437070223, c_{S2,D3} = 144.52609291614627, c_{S2,D4} = 11.420133077898234, c_{S2,D5} = 153.1756819927813
- From S3: c_{S3,D1} = 9.674584301671008, c_{S3,D2} = 2.6191650959687944, c_{S3,D3} = 100.8242249168735, c_{S3,D4} = 3.2121910887916876, c_{S3,D5} = 133.8493396124168
- From S4: c_{S4,D1} = 270.57498480010247, c_{S4,D2} = 32.50253586, c_{S4,D3} = 4.6842098096469815, c_{S4,D4} = 1.5682269686546804, c_{S4,D5} = 9.58927599
- From S5: c_{S5,D1} = 226.0331910675782, c_{S5,D2} = 8.669161980826471, c_{S5,D3} = 65.47681316968448, c_{S5,D4} = 9.068765258459958, c_{S5,D5} = 202.65015316425533

Variables:
- x_{ij} ≥ 0, ∀ i ∈ S, j ∈ D

Objective:
Minimize total transportation cost:
\[
\min \Bigg[
269.3910588020795\,x_{S1,D1} + 1.4537335390933939\,x_{S1,D2} + 99.60345345756605\,x_{S1,D3} + 26.64078166309837\,x_{S1,D4} + 9.537688956880922\,x_{S1,D5}
\]
\[
+ 9.291846876785183\,x_{S2,D1} + 10.874778437070223\,x_{S2,D2} + 144.52609291614627\,x_{S2,D3} + 11.420133077898234\,x_{S2,D4} + 153.1756819927813\,x_{S2,D5}
\]
\[
+ 9.674584301671008\,x_{S3,D1} + 2.6191650959687944\,x_{S3,D2} + 100.8242249168735\,x_{S3,D3} + 3.2121910887916876\,x_{S3,D4} + 133.8493396124168\,x_{S3,D5}
\]
\[
+ 270.57498480010247\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.6842098096469815\,x_{S4,D3} + 1.5682269686546804\,x_{S4,D4} + 9.58927599\,x_{S4,D5}
\]
\[
+ 226.0331910675782\,x_{S5,D1} + 8.669161980826471\,x_{S5,D2} + 65.47681316968448\,x_{S5,D3} + 9.068765258459958\,x_{S5,D4} + 202.65015316425533\,x_{S5,D5}
\Bigg]
\]

Subject to:

Demand satisfaction (for each store):
\[
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} = 428
\]
\[
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} = 217
\]
\[
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} = 214
\]
\[
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} = 380
\]
\[
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} = 254
\]

Supply capacity (for each warehouse):
\[
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} \leq 428
\]
\[
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} \leq 217
\]
\[
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} \leq 214
\]
\[
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} \leq 380
\]
\[
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} \leq 254
\]

Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in \{S1, S2, S3, S4, S5\},\ j \in \{D1, D2, D3, D4, D5\}
\]

This is the complete numerical formulation of the GreenMart transportation problem, preserving all source-ordered identifiers and coefficients.