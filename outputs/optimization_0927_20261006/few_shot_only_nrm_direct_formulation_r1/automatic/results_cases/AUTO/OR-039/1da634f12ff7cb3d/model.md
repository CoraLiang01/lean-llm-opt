Let  
- $x_{ij}$ = number of units of vehicle type $i$ stored in warehouse $j$, where $i$ indexes ProductName from products.csv and $j$ indexes Warehouse ID from capacity.csv.  
- $v_i$ = Value of vehicle type $i$ (from products.csv)  
- $w_i$ = Weight of vehicle type $i$ (from products.csv)  
- $C_j$ = Capacity of warehouse $j$ (from capacity.csv)  

**Indices:**  
- $i \in$ {Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks}  
- $j \in$ {Warehouse 1, Warehouse 2, Warehouse 3, Warehouse 4, Warehouse 5, Warehouse 6, Warehouse 7, Warehouse 8, Warehouse 9, Warehouse 10}  

**Parameters:**  
From products.csv:  
| ProductName           | Value | Weight |
|-----------------------|-------|--------|
| Sedans                | 1200  | 20     |
| SUVs                  | 1800  | 15     |
| Electric Vehicles     | 2500  | 25     |
| Hybrid Vehicles       | 2000  | 18     |
| Trucks                | 1500  | 10     |
| Sports Cars           | 3000  | 5      |
| Compact Cars          | 1000  | 22     |
| Luxury Sedans         | 3500  | 8      |
| Vans                  | 1600  | 12     |
| Pickup Trucks         | 1700  | 7      |

From capacity.csv:  
| Warehouse ID   | Capacity |
|----------------|----------|
| Warehouse 1    | 100      |
| Warehouse 2    | 80       |
| Warehouse 3    | 120      |
| Warehouse 4    | 90       |
| Warehouse 5    | 50       |
| Warehouse 6    | 30       |
| Warehouse 7    | 110      |
| Warehouse 8    | 40       |
| Warehouse 9    | 60       |
| Warehouse 10   | 35       |

**Decision Variables:**  
- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i, j$

**Objective:**  
Maximize total value of vehicles stored:
$$
\max \sum_{i} \sum_{j} v_i \cdot x_{ij}
$$

**Constraints:**  

1. **Warehouse Capacity Constraints:**  
For each warehouse $j$,
$$
\sum_{i} w_i \cdot x_{ij} \leq C_j \qquad \forall j
$$

2. **Non-negativity and Integrality:**  
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
$$

**Explicitly, the model is:**

Maximize
\[
1200 \sum_{j} x_{\text{Sedans},j}
+ 1800 \sum_{j} x_{\text{SUVs},j}
+ 2500 \sum_{j} x_{\text{Electric Vehicles},j}
+ 2000 \sum_{j} x_{\text{Hybrid Vehicles},j}
+ 1500 \sum_{j} x_{\text{Trucks},j}
+ 3000 \sum_{j} x_{\text{Sports Cars},j}
+ 1000 \sum_{j} x_{\text{Compact Cars},j}
+ 3500 \sum_{j} x_{\text{Luxury Sedans},j}
+ 1600 \sum_{j} x_{\text{Vans},j}
+ 1700 \sum_{j} x_{\text{Pickup Trucks},j}
\]

Subject to, for each warehouse $j$:

- For Warehouse 1:
  \[
  20x_{\text{Sedans},1} + 15x_{\text{SUVs},1} + 25x_{\text{Electric Vehicles},1} + 18x_{\text{Hybrid Vehicles},1} + 10x_{\text{Trucks},1} + 5x_{\text{Sports Cars},1} + 22x_{\text{Compact Cars},1} + 8x_{\text{Luxury Sedans},1} + 12x_{\text{Vans},1} + 7x_{\text{Pickup Trucks},1} \leq 100
  \]
- For Warehouse 2:
  \[
  20x_{\text{Sedans},2} + 15x_{\text{SUVs},2} + 25x_{\text{Electric Vehicles},2} + 18x_{\text{Hybrid Vehicles},2} + 10x_{\text{Trucks},2} + 5x_{\text{Sports Cars},2} + 22x_{\text{Compact Cars},2} + 8x_{\text{Luxury Sedans},2} + 12x_{\text{Vans},2} + 7x_{\text{Pickup Trucks},2} \leq 80
  \]
- For Warehouse 3:
  \[
  20x_{\text{Sedans},3} + 15x_{\text{SUVs},3} + 25x_{\text{Electric Vehicles},3} + 18x_{\text{Hybrid Vehicles},3} + 10x_{\text{Trucks},3} + 5x_{\text{Sports Cars},3} + 22x_{\text{Compact Cars},3} + 8x_{\text{Luxury Sedans},3} + 12x_{\text{Vans},3} + 7x_{\text{Pickup Trucks},3} \leq 120
  \]
- For Warehouse 4:
  \[
  20x_{\text{Sedans},4} + 15x_{\text{SUVs},4} + 25x_{\text{Electric Vehicles},4} + 18x_{\text{Hybrid Vehicles},4} + 10x_{\text{Trucks},4} + 5x_{\text{Sports Cars},4} + 22x_{\text{Compact Cars},4} + 8x_{\text{Luxury Sedans},4} + 12x_{\text{Vans},4} + 7x_{\text{Pickup Trucks},4} \leq 90
  \]
- For Warehouse 5:
  \[
  20x_{\text{Sedans},5} + 15x_{\text{SUVs},5} + 25x_{\text{Electric Vehicles},5} + 18x_{\text{Hybrid Vehicles},5} + 10x_{\text{Trucks},5} + 5x_{\text{Sports Cars},5} + 22x_{\text{Compact Cars},5} + 8x_{\text{Luxury Sedans},5} + 12x_{\text{Vans},5} + 7x_{\text{Pickup Trucks},5} \leq 50
  \]
- For Warehouse 6:
  \[
  20x_{\text{Sedans},6} + 15x_{\text{SUVs},6} + 25x_{\text{Electric Vehicles},6} + 18x_{\text{Hybrid Vehicles},6} + 10x_{\text{Trucks},6} + 5x_{\text{Sports Cars},6} + 22x_{\text{Compact Cars},6} + 8x_{\text{Luxury Sedans},6} + 12x_{\text{Vans},6} + 7x_{\text{Pickup Trucks},6} \leq 30
  \]
- For Warehouse 7:
  \[
  20x_{\text{Sedans},7} + 15x_{\text{SUVs},7} + 25x_{\text{Electric Vehicles},7} + 18x_{\text{Hybrid Vehicles},7} + 10x_{\text{Trucks},7} + 5x_{\text{Sports Cars},7} + 22x_{\text{Compact Cars},7} + 8x_{\text{Luxury Sedans},7} + 12x_{\text{Vans},7} + 7x_{\text{Pickup Trucks},7} \leq 110
  \]
- For Warehouse 8:
  \[
  20x_{\text{Sedans},8} + 15x_{\text{SUVs},8} + 25x_{\text{Electric Vehicles},8} + 18x_{\text{Hybrid Vehicles},8} + 10x_{\text{Trucks},8} + 5x_{\text{Sports Cars},8} + 22x_{\text{Compact Cars},8} + 8x_{\text{Luxury Sedans},8} + 12x_{\text{Vans},8} + 7x_{\text{Pickup Trucks},8} \leq 40
  \]
- For Warehouse 9:
  \[
  20x_{\text{Sedans},9} + 15x_{\text{SUVs},9} + 25x_{\text{Electric Vehicles},9} + 18x_{\text{Hybrid Vehicles},9} + 10x_{\text{Trucks},9} + 5x_{\text{Sports Cars},9} + 22x_{\text{Compact Cars},9} + 8x_{\text{Luxury Sedans},9} + 12x_{\text{Vans},9} + 7x_{\text{Pickup Trucks},9} \leq 60
  \]
- For Warehouse 10:
  \[
  20x_{\text{Sedans},10} + 15x_{\text{SUVs},10} + 25x_{\text{Electric Vehicles},10} + 18x_{\text{Hybrid Vehicles},10} + 10x_{\text{Trucks},10} + 5x_{\text{Sports Cars},10} + 22x_{\text{Compact Cars},10} + 8x_{\text{Luxury Sedans},10} + 12x_{\text{Vans},10} + 7x_{\text{Pickup Trucks},10} \leq 35
  \]

And for all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

Where $x_{ij}$ is the number of units of vehicle type $i$ stored in warehouse $j$.