{
  "query": "A boat dealership needs to assign different types of boats (e.g. speedboats, fishing boats, catamarans, etc.) to different display areas. Specifically, the boat dealer has several display areas, and the capacity limit for each display area is provided in ‘capacity.csv’. Predefined values and dimensions for each vessel type can be found in ‘products.csv’. Our goal is to determine the optimal number of each vessel type to place in each showcase to maximise the total value of the vessels in all showcases, while ensuring that the total size of the vessels in each showcase does not exceed its capacity. The decision variable x_{ij} denotes the number of vessels of type j to be placed in display area i. The decision variable x_{ij} is the number of vessels of type j to be placed in display area i.",
  "relationships": [],
  "tables": [
    {
      "columns": [
        "DisplayID",
        "Capacity"
      ],
      "file_name": "capacity.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "Capacity": "457",
            "DisplayID": "1"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "Capacity": "604",
            "DisplayID": "2"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "Capacity": "751",
            "DisplayID": "3"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "Capacity": "468",
            "DisplayID": "4"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "Capacity": "343",
            "DisplayID": "5"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "Capacity": "408",
            "DisplayID": "6"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "Capacity": "741",
            "DisplayID": "7"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "Capacity": "914",
            "DisplayID": "8"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "Capacity": "682",
            "DisplayID": "9"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "Capacity": "409",
            "DisplayID": "10"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "Capacity": "342",
            "DisplayID": "11"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "Capacity": "903",
            "DisplayID": "12"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "Capacity": "680",
            "DisplayID": "13"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "Capacity": "886",
            "DisplayID": "14"
          }
        }
      ],
      "table_id": "file_0_view_0"
    },
    {
      "columns": [
        "ProductName",
        "Value",
        "Weight"
      ],
      "file_name": "products.csv",
      "records": [
        {
          "data_row_number": 0,
          "values": {
            "ProductName": "Speedboat",
            "Value": "29664",
            "Weight": "18"
          }
        },
        {
          "data_row_number": 1,
          "values": {
            "ProductName": "Fishing Boat",
            "Value": "31778",
            "Weight": "36"
          }
        },
        {
          "data_row_number": 2,
          "values": {
            "ProductName": "Catamaran",
            "Value": "73501",
            "Weight": "25"
          }
        },
        {
          "data_row_number": 3,
          "values": {
            "ProductName": "Yacht",
            "Value": "78255",
            "Weight": "16"
          }
        },
        {
          "data_row_number": 4,
          "values": {
            "ProductName": "Sailboat",
            "Value": "93606",
            "Weight": "97"
          }
        },
        {
          "data_row_number": 5,
          "values": {
            "ProductName": "Kayak",
            "Value": "46983",
            "Weight": "35"
          }
        },
        {
          "data_row_number": 6,
          "values": {
            "ProductName": "Canoe",
            "Value": "95026",
            "Weight": "32"
          }
        },
        {
          "data_row_number": 7,
          "values": {
            "ProductName": "Houseboat",
            "Value": "57685",
            "Weight": "100"
          }
        },
        {
          "data_row_number": 8,
          "values": {
            "ProductName": "Pontoon",
            "Value": "60323",
            "Weight": "43"
          }
        },
        {
          "data_row_number": 9,
          "values": {
            "ProductName": "Jet Ski",
            "Value": "91224",
            "Weight": "15"
          }
        },
        {
          "data_row_number": 10,
          "values": {
            "ProductName": "Rowboat",
            "Value": "44003",
            "Weight": "95"
          }
        },
        {
          "data_row_number": 11,
          "values": {
            "ProductName": "Hovercraft",
            "Value": "75998",
            "Weight": "57"
          }
        },
        {
          "data_row_number": 12,
          "values": {
            "ProductName": "Cabin Cruiser",
            "Value": "84525",
            "Weight": "13"
          }
        },
        {
          "data_row_number": 13,
          "values": {
            "ProductName": "Wakeboard Boat",
            "Value": "66207",
            "Weight": "44"
          }
        },
        {
          "data_row_number": 14,
          "values": {
            "ProductName": "Dinghy",
            "Value": "65002",
            "Weight": "64"
          }
        },
        {
          "data_row_number": 15,
          "values": {
            "ProductName": "Trawler",
            "Value": "33132",
            "Weight": "88"
          }
        },
        {
          "data_row_number": 16,
          "values": {
            "ProductName": "Paddle Boat",
            "Value": "69239",
            "Weight": "42"
          }
        },
        {
          "data_row_number": 17,
          "values": {
            "ProductName": "Submarine",
            "Value": "66948",
            "Weight": "46"
          }
        },
        {
          "data_row_number": 18,
          "values": {
            "ProductName": "RIB",
            "Value": "88240",
            "Weight": "24"
          }
        },
        {
          "data_row_number": 19,
          "values": {
            "ProductName": "Skiff",
            "Value": "48858",
            "Weight": "93"
          }
        }
      ],
      "table_id": "file_1_view_0"
    }
  ]
}
