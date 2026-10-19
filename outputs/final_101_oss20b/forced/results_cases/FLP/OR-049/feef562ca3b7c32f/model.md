##### Decision Variables

$x_{ij}\in \mathbb{Z}_{\ge 0}$: number of units of product $j\in J$ placed on shelf $i\in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij}$

##### Constraints

1. Shelf capacity: $\displaystyle \sum_{j\in J} w_j\,x_{ij}\leq c_i,\quad \forall i\in I$  
2. Non‑negativity and integrality: $\displaystyle x_{ij}\in \mathbb{Z}_{\ge 0},\quad \forall i\in I,\;j\in J$

##### Sets and Parameters

$I=\{1,2,3,4,5,6,7,8,9,10\}$ (shelf indices)  
$J=\{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\}$ (product indices)

Capacity vector $c=[5.0,\;7.0,\;6.0,\;8.0,\;5.5,\;9.0,\;6.5,\;7.5,\;8.2,\;5.7]$

Product names vector  
$P=\{\text{Smartphone},\;\text{Laptop},\;\text{Headphones},\;\text{Camera},\;\text{Smartwatch},\;\text{Tablet},\;\text{Bluetooth Speaker},\;\text{Keyboard},\;\text{Mouse},\;\text{Monitor},\;\text{Printer},\;\text{External Hard Drive},\;\text{Router},\;\text{Power Bank},\;\text{Memory Card},\;\text{USB Flash Drive},\;\text{Smart Home Hub},\;\text{Gaming Console},\;\text{Fitness Tracker},\;\text{E-Reader}\}$

Value vector $v=[200,\;1500,\;100,\;800,\;250,\;600,\;150,\;80,\;50,\;300,\;400,\;120,\;60,\;40,\;30,\;25,\;100,\;500,\;90,\;180]$

Weight vector $w=[1.0,\;5.0,\;0.5,\;2.0,\;0.3,\;1.5,\;1.0,\;0.8,\;0.2,\;3.0,\;4.0,\;0.5,\;0.3,\;0.4,\;0.05,\;0.02,\;0.6,\;4.0,\;0.2,\;0.5]$

##### Retrieved Information

- **Shelves (capacity.csv)**  
  ShelfID: 1, Capacity 5.0  
  ShelfID: 2, Capacity 7.0  
  ShelfID: 3, Capacity 6.0  
  ShelfID: 4, Capacity 8.0  
  ShelfID: 5, Capacity 5.5  
  ShelfID: 6, Capacity 9.0  
  ShelfID: 7, Capacity 6.5  
  ShelfID: 8, Capacity 7.5  
  ShelfID: 9, Capacity 8.2  
  ShelfID: 10, Capacity 5.7  

- **Products (products.csv)**  
  ProductName, Value, Weight  
  Smartphone, 200, 1.0  
  Laptop, 1500, 5.0  
  Headphones, 100, 0.5  
  Camera, 800, 2.0  
  Smartwatch, 250, 0.3  
  Tablet, 600, 1.5  
  Bluetooth Speaker, 150, 1.0  
  Keyboard, 80, 0.8  
  Mouse, 50, 0.2  
  Monitor, 300, 3.0  
  Printer, 400, 4.0  
  External Hard Drive, 120, 0.5  
  Router, 60, 0.3  
  Power Bank, 40, 0.4  
  Memory Card, 30, 0.05  
  USB Flash Drive, 25, 0.02  
  Smart Home Hub, 100, 0.6  
  Gaming Console, 500, 4.0  
  Fitness Tracker, 90, 0.2  
  E-Reader, 180, 0.5