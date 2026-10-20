##### Objective Function: 

$\quad \quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}$

##### Constraints

###### 1. Capacity Constraints: 

$\quad \quad \sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i , \quad \forall i=1,\dots,10$

###### 2. Minimum Quantity of First Product: 

$\quad \quad \sum_{i=1}^{10} x_{i1} \geq 5$

###### 3. Non-negativity and Integrality Constraints: 

$\quad \quad x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\dots,10,\; \forall j=1,\dots,20$

###### Retrieved Information

{
  "items": [
    "Smartphone",
    "Laptop",
    "Headphones",
    "Camera",
    "Smartwatch",
    "Tablet",
    "Bluetooth Speaker",
    "Keyboard",
    "Mouse",
    "Monitor",
    "Printer",
    "External Hard Drive",
    "Router",
    "Power Bank",
    "Memory Card",
    "USB Flash Drive",
    "Smart Home Hub",
    "Gaming Console",
    "Fitness Tracker",
    "E-Reader"
  ],
  "profit": {
    "Smartphone": 200,
    "Laptop": 1500,
    "Headphones": 100,
    "Camera": 800,
    "Smartwatch": 250,
    "Tablet": 600,
    "Bluetooth Speaker": 150,
    "Keyboard": 80,
    "Mouse": 50,
    "Monitor": 300,
    "Printer": 400,
    "External Hard Drive": 120,
    "Router": 60,
    "Power Bank": 40,
    "Memory Card": 30,
    "USB Flash Drive": 25,
    "Smart Home Hub": 100,
    "Gaming Console": 500,
    "Fitness Tracker": 90,
    "E-Reader": 180
  },
  "shelves": [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
  ],
  "capacity": {
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
  "weight": {
    "Smartphone": 1.0,
    "Laptop": 5.0,
    "Headphones": 0.5,
    "Camera": 2.0,
    "Smartwatch": 0.3,
    "Tablet": 1.5,
    "Bluetooth Speaker": 1.0,
    "Keyboard": 0.8,
    "Mouse": 0.2,
    "Monitor": 3.0,
    "Printer": 4.0,
    "External Hard Drive": 0.5,
    "Router": 0.3,
    "Power Bank": 0.4,
    "Memory Card": 0.05,
    "USB Flash Drive": 0.02,
    "Smart Home Hub": 0.6,
    "Gaming Console": 4.0,
    "Fitness Tracker": 0.2,
    "E-Reader": 0.5
  }
}