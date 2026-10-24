{
  "query": "A regional distributor can ship goods from depots to markets. Depot capacities are listed in depot_capacity.csv, market demands are listed in market_demand.csv, per-unit route costs are listed in route_variable_costs.csv, and fixed route activation costs are listed in route_fixed_costs.csv. A route may carry shipments only if it is activated.\n\nFormulate a minimum-cost fixed-charge transportation model. For each depot-market route i-j, define x_ij as the nonnegative shipment quantity and y_ij as a binary variable equal to 1 if route i-j is activated. The objective is to minimize variable shipping cost plus fixed route activation cost. The model should include market demand constraints, depot supply upper-bound constraints, shipment-to-route activation linking constraints using M_ij = min(depot capacity_i, market demand_j), nonnegativity constraints for shipment variables, and binary restrictions for route variables.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "Depot",
        "SupplyCapacity"
      ],
      "file_name": "depot_capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Depot": "D1",
            "SupplyCapacity": "120"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Depot": "D2",
            "SupplyCapacity": "100"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Depot": "D3",
            "SupplyCapacity": "140"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Depot": "D4",
            "SupplyCapacity": "90"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "Market",
        "Demand"
      ],
      "file_name": "market_demand.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Demand": "65",
            "Market": "M1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Demand": "80",
            "Market": "M2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Demand": "75",
            "Market": "M3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Demand": "95",
            "Market": "M4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Demand": "70",
            "Market": "M5"
          }
        }
      ],
      "table_id": "file_1_view_0"
    },
    {
      "columns": [
        "Depot",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5"
      ],
      "file_name": "route_variable_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Depot": "D1",
            "M1": "4",
            "M2": "5",
            "M3": "17",
            "M4": "18",
            "M5": "16"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Depot": "D2",
            "M1": "15",
            "M2": "14",
            "M3": "3",
            "M4": "6",
            "M5": "17"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Depot": "D3",
            "M1": "18",
            "M2": "16",
            "M3": "15",
            "M4": "4",
            "M5": "5"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Depot": "D4",
            "M1": "6",
            "M2": "7",
            "M3": "14",
            "M4": "16",
            "M5": "13"
          }
        }
      ],
      "table_id": "file_2_view_0"
    },
    {
      "columns": [
        "Depot",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5"
      ],
      "file_name": "route_fixed_costs.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Depot": "D1",
            "M1": "240",
            "M2": "270",
            "M3": "560",
            "M4": "590",
            "M5": "540"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Depot": "D2",
            "M1": "520",
            "M2": "500",
            "M3": "230",
            "M4": "280",
            "M5": "570"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Depot": "D3",
            "M1": "610",
            "M2": "580",
            "M3": "540",
            "M4": "260",
            "M5": "250"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Depot": "D4",
            "M1": "290",
            "M2": "320",
            "M3": "500",
            "M4": "550",
            "M5": "480"
          }
        }
      ],
      "table_id": "file_3_view_0"
    }
  ]
}
