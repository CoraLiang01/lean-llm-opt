##### Sets and Indices

- Let $W = \{\text{Warehouse 1}, \text{Warehouse 2}, \text{Warehouse 3}, \text{Warehouse 4}, \text{Warehouse 5}, \text{Warehouse 6}, \text{Warehouse 7}, \text{Warehouse 8}, \text{Warehouse 9}, \text{Warehouse 10}\}$ be the set of warehouses, indexed by $w$.
- Let $P = \{\text{Sedans}, \text{SUVs}, \text{Electric Vehicles}, \text{Hybrid Vehicles}, \text{Trucks}, \text{Sports Cars}, \text{Compact Cars}, \text{Luxury Sedans}, \text{Vans}, \text{Pickup Trucks}\}$ be the set of vehicle types, indexed by $p$.

##### Parameters

- $C_w$: Capacity of warehouse $w$.
    - $C_{\text{Warehouse 1}} = 100$
    - $C_{\text{Warehouse 2}} = 80$
    - $C_{\text{Warehouse 3}} = 120$
    - $C_{\text{Warehouse 4}} = 90$
    - $C_{\text{Warehouse 5}} = 50$
    - $C_{\text{Warehouse 6}} = 30$
    - $C_{\text{Warehouse 7}} = 110$
    - $C_{\text{Warehouse 8}} = 40$
    - $C_{\text{Warehouse 9}} = 60$
    - $C_{\text{Warehouse 10}} = 35$
- $v_p$: Value (benefit coefficient) per unit of vehicle type $p$.
    - $v_{\text{Sedans}} = 1200$
    - $v_{\text{SUVs}} = 1800$
    - $v_{\text{Electric Vehicles}} = 2500$
    - $v_{\text{Hybrid Vehicles}} = 2000$
    - $v_{\text{Trucks}} = 1500$
    - $v_{\text{Sports Cars}} = 3000$
    - $v_{\text{Compact Cars}} = 1000$
    - $v_{\text{Luxury Sedans}} = 3500$
    - $v_{\text{Vans}} = 1600$
    - $v_{\text{Pickup Trucks}} = 1700$
- $w_p$: Weight (storage space required) per unit of vehicle type $p$.
    - $w_{\text{Sedans}} = 20$
    - $w_{\text{SUVs}} = 15$
    - $w_{\text{Electric Vehicles}} = 25$
    - $w_{\text{Hybrid Vehicles}} = 18$
    - $w_{\text{Trucks}} = 10$
    - $w_{\text{Sports Cars}} = 5$
    - $w_{\text{Compact Cars}} = 22$
    - $w_{\text{Luxury Sedans}} = 8$
    - $w_{\text{Vans}} = 12$
    - $w_{\text{Pickup Trucks}} = 7$

##### Decision Variables

Let $x_{w,p} \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $p$ to store in warehouse $w$ (integer, nonnegative).

##### Objective Function

\[
\max \sum_{w \in W} \sum_{p \in P} v_p \, x_{w,p}
\]

##### Constraints

1. Warehouse capacity constraints (for each warehouse $w$):

\[
\sum_{p \in P} w_p \, x_{w,p} \leq C_w \qquad \forall w \in W
\]

2. Nonnegativity and integrality:

\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \qquad \forall w \in W,\, p \in P
\]

##### Complete Numerical Formulation

Let $x_{w,p}$ be the integer number of vehicles of type $p$ stored in warehouse $w$.

\[
\max \left(
\sum_{w \in W} \sum_{p \in P} v_p \, x_{w,p}
\right)
\]

subject to, for each warehouse:

- Warehouse 1: $20x_{\text{Warehouse 1},\text{Sedans}} + 15x_{\text{Warehouse 1},\text{SUVs}} + 25x_{\text{Warehouse 1},\text{Electric Vehicles}} + 18x_{\text{Warehouse 1},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 1},\text{Trucks}} + 5x_{\text{Warehouse 1},\text{Sports Cars}} + 22x_{\text{Warehouse 1},\text{Compact Cars}} + 8x_{\text{Warehouse 1},\text{Luxury Sedans}} + 12x_{\text{Warehouse 1},\text{Vans}} + 7x_{\text{Warehouse 1},\text{Pickup Trucks}} \leq 100$
- Warehouse 2: $20x_{\text{Warehouse 2},\text{Sedans}} + 15x_{\text{Warehouse 2},\text{SUVs}} + 25x_{\text{Warehouse 2},\text{Electric Vehicles}} + 18x_{\text{Warehouse 2},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 2},\text{Trucks}} + 5x_{\text{Warehouse 2},\text{Sports Cars}} + 22x_{\text{Warehouse 2},\text{Compact Cars}} + 8x_{\text{Warehouse 2},\text{Luxury Sedans}} + 12x_{\text{Warehouse 2},\text{Vans}} + 7x_{\text{Warehouse 2},\text{Pickup Trucks}} \leq 80$
- Warehouse 3: $20x_{\text{Warehouse 3},\text{Sedans}} + 15x_{\text{Warehouse 3},\text{SUVs}} + 25x_{\text{Warehouse 3},\text{Electric Vehicles}} + 18x_{\text{Warehouse 3},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 3},\text{Trucks}} + 5x_{\text{Warehouse 3},\text{Sports Cars}} + 22x_{\text{Warehouse 3},\text{Compact Cars}} + 8x_{\text{Warehouse 3},\text{Luxury Sedans}} + 12x_{\text{Warehouse 3},\text{Vans}} + 7x_{\text{Warehouse 3},\text{Pickup Trucks}} \leq 120$
- Warehouse 4: $20x_{\text{Warehouse 4},\text{Sedans}} + 15x_{\text{Warehouse 4},\text{SUVs}} + 25x_{\text{Warehouse 4},\text{Electric Vehicles}} + 18x_{\text{Warehouse 4},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 4},\text{Trucks}} + 5x_{\text{Warehouse 4},\text{Sports Cars}} + 22x_{\text{Warehouse 4},\text{Compact Cars}} + 8x_{\text{Warehouse 4},\text{Luxury Sedans}} + 12x_{\text{Warehouse 4},\text{Vans}} + 7x_{\text{Warehouse 4},\text{Pickup Trucks}} \leq 90$
- Warehouse 5: $20x_{\text{Warehouse 5},\text{Sedans}} + 15x_{\text{Warehouse 5},\text{SUVs}} + 25x_{\text{Warehouse 5},\text{Electric Vehicles}} + 18x_{\text{Warehouse 5},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 5},\text{Trucks}} + 5x_{\text{Warehouse 5},\text{Sports Cars}} + 22x_{\text{Warehouse 5},\text{Compact Cars}} + 8x_{\text{Warehouse 5},\text{Luxury Sedans}} + 12x_{\text{Warehouse 5},\text{Vans}} + 7x_{\text{Warehouse 5},\text{Pickup Trucks}} \leq 50$
- Warehouse 6: $20x_{\text{Warehouse 6},\text{Sedans}} + 15x_{\text{Warehouse 6},\text{SUVs}} + 25x_{\text{Warehouse 6},\text{Electric Vehicles}} + 18x_{\text{Warehouse 6},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 6},\text{Trucks}} + 5x_{\text{Warehouse 6},\text{Sports Cars}} + 22x_{\text{Warehouse 6},\text{Compact Cars}} + 8x_{\text{Warehouse 6},\text{Luxury Sedans}} + 12x_{\text{Warehouse 6},\text{Vans}} + 7x_{\text{Warehouse 6},\text{Pickup Trucks}} \leq 30$
- Warehouse 7: $20x_{\text{Warehouse 7},\text{Sedans}} + 15x_{\text{Warehouse 7},\text{SUVs}} + 25x_{\text{Warehouse 7},\text{Electric Vehicles}} + 18x_{\text{Warehouse 7},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 7},\text{Trucks}} + 5x_{\text{Warehouse 7},\text{Sports Cars}} + 22x_{\text{Warehouse 7},\text{Compact Cars}} + 8x_{\text{Warehouse 7},\text{Luxury Sedans}} + 12x_{\text{Warehouse 7},\text{Vans}} + 7x_{\text{Warehouse 7},\text{Pickup Trucks}} \leq 110$
- Warehouse 8: $20x_{\text{Warehouse 8},\text{Sedans}} + 15x_{\text{Warehouse 8},\text{SUVs}} + 25x_{\text{Warehouse 8},\text{Electric Vehicles}} + 18x_{\text{Warehouse 8},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 8},\text{Trucks}} + 5x_{\text{Warehouse 8},\text{Sports Cars}} + 22x_{\text{Warehouse 8},\text{Compact Cars}} + 8x_{\text{Warehouse 8},\text{Luxury Sedans}} + 12x_{\text{Warehouse 8},\text{Vans}} + 7x_{\text{Warehouse 8},\text{Pickup Trucks}} \leq 40$
- Warehouse 9: $20x_{\text{Warehouse 9},\text{Sedans}} + 15x_{\text{Warehouse 9},\text{SUVs}} + 25x_{\text{Warehouse 9},\text{Electric Vehicles}} + 18x_{\text{Warehouse 9},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 9},\text{Trucks}} + 5x_{\text{Warehouse 9},\text{Sports Cars}} + 22x_{\text{Warehouse 9},\text{Compact Cars}} + 8x_{\text{Warehouse 9},\text{Luxury Sedans}} + 12x_{\text{Warehouse 9},\text{Vans}} + 7x_{\text{Warehouse 9},\text{Pickup Trucks}} \leq 60$
- Warehouse 10: $20x_{\text{Warehouse 10},\text{Sedans}} + 15x_{\text{Warehouse 10},\text{SUVs}} + 25x_{\text{Warehouse 10},\text{Electric Vehicles}} + 18x_{\text{Warehouse 10},\text{Hybrid Vehicles}} + 10x_{\text{Warehouse 10},\text{Trucks}} + 5x_{\text{Warehouse 10},\text{Sports Cars}} + 22x_{\text{Warehouse 10},\text{Compact Cars}} + 8x_{\text{Warehouse 10},\text{Luxury Sedans}} + 12x_{\text{Warehouse 10},\text{Vans}} + 7x_{\text{Warehouse 10},\text{Pickup Trucks}} \leq 35$

and

\[
x_{w,p} \in \mathbb{Z}_{\geq 0} \qquad \forall w \in W,\, p \in P
\]