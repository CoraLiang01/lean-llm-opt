{
  "query": "A snack distributor ships cartons from plants to regional retailers. Plant capacities are listed in plant_capacity.csv, retailer demands are listed in retailer_demand.csv, variable per-carton route costs are listed in route_variable_costs.csv, and fixed route activation costs are listed in route_fixed_costs.csv. A route can carry flow only if it is activated.\n\nFormulate a minimum-cost fixed-charge transportation model. For each plant-retailer route i-j, define x_ij as the nonnegative shipment quantity and y_ij as a binary variable equal to 1 if route i-j is opened. The objective is to minimize variable transportation cost plus fixed route-opening cost. The model should include retailer demand constraints, plant supply upper-bound constraints, shipment-to-route activation linking constraints using M_ij = min(plant capacity_i, retailer demand_j), nonnegativity constraints for shipment variables, and binary restrictions for route variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Plant",
        "SupplyCapacity"
      ],
      "file_name": "plant_capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Plant": "P1",
            "SupplyCapacity": "190"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Plant": "P2",
            "SupplyCapacity": "160"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Plant": "P3",
            "SupplyCapacity": "150"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Retailer",
        "Demand"
      ],
      "file_name": "retailer_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Demand": "70",
            "Retailer": "R1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Demand": "85",
            "Retailer": "R2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Demand": "75",
            "Retailer": "R3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Demand": "65",
            "Retailer": "R4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Demand": "95",
            "Retailer": "R5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Demand": "80",
            "Retailer": "R6"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "Plant",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6"
      ],
      "file_name": "route_variable_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Plant": "P1",
            "R1": "3",
            "R2": "4",
            "R3": "18",
            "R4": "20",
            "R5": "22",
            "R6": "19"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Plant": "P2",
            "R1": "17",
            "R2": "16",
            "R3": "4",
            "R4": "5",
            "R5": "20",
            "R6": "18"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Plant": "P3",
            "R1": "21",
            "R2": "19",
            "R3": "18",
            "R4": "17",
            "R5": "3",
            "R6": "4"
          }
        }
      ],
      "table_id": "file_2_view_0"
    },
    {
      "columns": [
        "Plant",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6"
      ],
      "file_name": "route_fixed_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Plant": "P1",
            "R1": "260",
            "R2": "300",
            "R3": "620",
            "R4": "670",
            "R5": "710",
            "R6": "660"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Plant": "P2",
            "R1": "610",
            "R2": "590",
            "R3": "280",
            "R4": "310",
            "R5": "690",
            "R6": "640"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Plant": "P3",
            "R1": "700",
            "R2": "650",
            "R3": "630",
            "R4": "600",
            "R5": "250",
            "R6": "290"
          }
        }
      ],
      "table_id": "file_3_view_0"
    }
  ]
}
