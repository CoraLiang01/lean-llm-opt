##### Objective Function: 

$\quad \quad \max \sum_{i} p_{i}\;x_{i}$  

##### Constraints  

###### 1. Demand Constraints:  

$\quad \quad x_{i}\;\leq\;D_{i},\;\;\forall i$  

###### 2. Inventory Constraints:  

$\quad \quad x_{i}\;\leq\;I_{i},\;\;\forall i$  

###### 3. Non‑negativity and Integrality Constraints:  

$\quad \quad x_{i}\;\in\;\mathbb{Z}_{\geq 0},\;\;\forall i$  

**Indices**  
$i\in\{1,2\}$ where  
- $i=1$ corresponds to product “27in 4K Gaming Monitor”  
- $i=2$ corresponds to product “27in FHD Monitor”  

**Retrieved Information**

```json
{
  "items": [
    "27in 4K Gaming Monitor",
    "27in FHD Monitor"
  ],
  "revenue": {
    "27in 4K Gaming Monitor": 389.99,
    "27in FHD Monitor": 149.99
  },
  "demand": {
    "27in 4K Gaming Monitor": 12474,
    "27in FHD Monitor": 15057
  },
  "inventory": {
    "27in 4K Gaming Monitor": 62440,
    "27in FHD Monitor": 75500
  }
}
```