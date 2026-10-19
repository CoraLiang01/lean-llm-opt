##### Objective Function:

$$ \min \sum_{s \in \text{Suppliers}} \sum_{c \in \text{Customers}} \text{cost}_{s,c} \cdot x_{s,c} $$

##### Constraints

###### 1. Demand Satisfaction Constraints:

$$ \sum_{s \in \text{Suppliers}} x_{s,c} = \text{demand}_c, \quad \forall c \in \text{Customers} $$

###### 2. Supply Capacity Constraints:

$$ \sum_{c \in \text{Customers}} x_{s,c} \leq \text{supply_capacity}_s, \quad \forall s \in \text{Suppliers} $$

###### 3. Non-negativity and Integrality Constraints:

$$ x_{s,c} \in \mathbb{Z}_{\geq 0}, \quad \forall s \in \text{Suppliers}, \forall c \in \text{Customers} $$

###### Retrieved Information

```json
{
  "suppliers": ["Supplier1","Supplier2","Supplier3","Supplier4","Supplier5"],
  "customers": ["Customer1","Customer2","Customer3","Customer4","Customer5","Customer6"],
  "demand": {"Customer1":70,"Customer2":80,"Customer3":60,"Customer4":90,"Customer5":85,"Customer6":95},
  "supply_capacity": {"Supplier1":200,"Supplier2":250,"Supplier3":230,"Supplier4":220,"Supplier5":210},
  "cost": {
    "Supplier1": {"Customer1":2,"Customer2":3,"Customer3":1,"Customer4":2,"Customer5":3,"Customer6":2},
    "Supplier2": {"Customer1":1,"Customer2":2,"Customer3":3,"Customer4":2,"Customer5":3,"Customer6":2},
    "Supplier3": {"Customer1":3,"Customer2":1,"Customer3":2,"Customer4":3,"Customer5":2,"Customer6":3},
    "Supplier4": {"Customer1":2,"Customer2":3,"Customer3":2,"Customer4":1,"Customer5":3,"Customer6":4},
    "Supplier5": {"Customer1":3,"Customer2":2,"Customer3":3,"Customer4":3,"Customer5":2,"Customer6":3}
  }
}
```