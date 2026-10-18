##### Sets and Indices

- Let $I = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$ be the set of displays (shelves), indexed by $i$.
- Let $J = \{$Smartphone, Laptop, Headphones, Camera, Smartwatch, Tablet, Bluetooth Speaker, Keyboard, Mouse, Monitor, Printer, External Hard Drive, Router, Power Bank, Memory Card, USB Flash Drive, Smart Home Hub, Gaming Console, Fitness Tracker, E-Reader$\}$ be the set of products, indexed by $j$.

##### Parameters

- $C_i$: Capacity of display $i$.

  $C_1 = 5.0$  
  $C_2 = 7.0$  
  $C_3 = 6.0$  
  $C_4 = 8.0$  
  $C_5 = 5.5$  
  $C_6 = 9.0$  
  $C_7 = 6.5$  
  $C_8 = 7.5$  
  $C_9 = 8.2$  
  $C_{10} = 5.7$

- $v_j$: Value of product $j$.

  - Smartphone: $v_{\text{Smartphone}} = 200$
  - Laptop: $v_{\text{Laptop}} = 1500$
  - Headphones: $v_{\text{Headphones}} = 100$
  - Camera: $v_{\text{Camera}} = 800$
  - Smartwatch: $v_{\text{Smartwatch}} = 250$
  - Tablet: $v_{\text{Tablet}} = 600$
  - Bluetooth Speaker: $v_{\text{Bluetooth Speaker}} = 150$
  - Keyboard: $v_{\text{Keyboard}} = 80$
  - Mouse: $v_{\text{Mouse}} = 50$
  - Monitor: $v_{\text{Monitor}} = 300$
  - Printer: $v_{\text{Printer}} = 400$
  - External Hard Drive: $v_{\text{External Hard Drive}} = 120$
  - Router: $v_{\text{Router}} = 60$
  - Power Bank: $v_{\text{Power Bank}} = 40$
  - Memory Card: $v_{\text{Memory Card}} = 30$
  - USB Flash Drive: $v_{\text{USB Flash Drive}} = 25$
  - Smart Home Hub: $v_{\text{Smart Home Hub}} = 100$
  - Gaming Console: $v_{\text{Gaming Console}} = 500$
  - Fitness Tracker: $v_{\text{Fitness Tracker}} = 90$
  - E-Reader: $v_{\text{E-Reader}} = 180$

- $w_j$: Weight of product $j$.

  - Smartphone: $w_{\text{Smartphone}} = 1.0$
  - Laptop: $w_{\text{Laptop}} = 5.0$
  - Headphones: $w_{\text{Headphones}} = 0.5$
  - Camera: $w_{\text{Camera}} = 2.0$
  - Smartwatch: $w_{\text{Smartwatch}} = 0.3$
  - Tablet: $w_{\text{Tablet}} = 1.5$
  - Bluetooth Speaker: $w_{\text{Bluetooth Speaker}} = 1.0$
  - Keyboard: $w_{\text{Keyboard}} = 0.8$
  - Mouse: $w_{\text{Mouse}} = 0.2$
  - Monitor: $w_{\text{Monitor}} = 3.0$
  - Printer: $w_{\text{Printer}} = 4.0$
  - External Hard Drive: $w_{\text{External Hard Drive}} = 0.5$
  - Router: $w_{\text{Router}} = 0.3$
  - Power Bank: $w_{\text{Power Bank}} = 0.4$
  - Memory Card: $w_{\text{Memory Card}} = 0.05$
  - USB Flash Drive: $w_{\text{USB Flash Drive}} = 0.02$
  - Smart Home Hub: $w_{\text{Smart Home Hub}} = 0.6$
  - Gaming Console: $w_{\text{Gaming Console}} = 4.0$
  - Fitness Tracker: $w_{\text{Fitness Tracker}} = 0.2$
  - E-Reader: $w_{\text{E-Reader}} = 0.5$

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on display $i$.

##### Objective Function

$\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}$

##### Constraints

1. **Display Capacity Constraints** (for each display $i$):

   $\sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I$

2. **Minimum Quantity of First Product (Smartphone) Across All Displays**:

   $\sum_{i \in I} x_{i,\text{Smartphone}} \geq 5$

3. **Nonnegativity and Integrality**:

   $x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I, \forall j \in J$

##### Retrieved Information

{
  "displays": {
    "1": 5.0,
    "2": 7.0,
    "3": 6.0,
    "4": 8.0,
    "5": 5.5,
    "6": 9.0,
    "7": 6.5,
    "8": 7.5,
    "9": 8.2,
    "10": 5.7
  },
  "products": {
    "Smartphone": {"Value": 200, "Weight": 1.0},
    "Laptop": {"Value": 1500, "Weight": 5.0},
    "Headphones": {"Value": 100, "Weight": 0.5},
    "Camera": {"Value": 800, "Weight": 2.0},
    "Smartwatch": {"Value": 250, "Weight": 0.3},
    "Tablet": {"Value": 600, "Weight": 1.5},
    "Bluetooth Speaker": {"Value": 150, "Weight": 1.0},
    "Keyboard": {"Value": 80, "Weight": 0.8},
    "Mouse": {"Value": 50, "Weight": 0.2},
    "Monitor": {"Value": 300, "Weight": 3.0},
    "Printer": {"Value": 400, "Weight": 4.0},
    "External Hard Drive": {"Value": 120, "Weight": 0.5},
    "Router": {"Value": 60, "Weight": 0.3},
    "Power Bank": {"Value": 40, "Weight": 0.4},
    "Memory Card": {"Value": 30, "Weight": 0.05},
    "USB Flash Drive": {"Value": 25, "Weight": 0.02},
    "Smart Home Hub": {"Value": 100, "Weight": 0.6},
    "Gaming Console": {"Value": 500, "Weight": 4.0},
    "Fitness Tracker": {"Value": 90, "Weight": 0.2},
    "E-Reader": {"Value": 180, "Weight": 0.5}
  }
}