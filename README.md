# 🌾 Humanitarian Food Supply Chain Optimization

This project models a **multi-stage supply chain** to deliver nutritious food commodities from suppliers to beneficiary camps, minimizing total costs while satisfying nutritional requirements. The optimization is implemented using **linear programming (LP) with Python and PuLP**.

---

## 📌 Project Overview

The goal is to optimize the flow of food commodities through a network of:

- **Suppliers** – supply various commodities
- **Ports** – handle incoming shipments and manage capacity
- **Warehouses** – store commodities before final delivery
- **Beneficiary Camps** – receive sufficient nutrients per month

The optimization considers:

- Procurement costs
- Sea transport costs
- Port and warehouse handling costs
- Land transport costs (port→warehouse→camp)
- Nutritional requirements for each camp

The model minimizes **total supply chain cost** while ensuring **all camps receive the required nutrients**.

---

## 🛠️ Tech Stack

- **Python** – data manipulation and LP modeling
- **Pandas** – reading Excel sheets and handling data
- **PuLP** – linear programming solver
- **CBC Solver** – default solver for PuLP

---

## 🧮 Data Description

- **Nodes** – Locations classified as Suppliers, Ports, Warehouses, Beneficiary Camps
- **Commodities** – Different food items
- **Nutrients** – Nutritional components (e.g., protein, vitamins)
- **Procurement** – Supplier prices for commodities
- **SeaTransport** – Shipping costs from suppliers to ports
- **LandTransport** – Costs from ports to warehouses and warehouses to camps
- **Nutritional values** – Nutrient content per commodity
- **Total nutrients per camp** – Required nutrient amounts for each camp

---
