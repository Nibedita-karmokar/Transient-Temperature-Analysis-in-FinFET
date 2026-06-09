In the Project_codes folder there are 2 python files (1) FDM_Main.py, and (2) Transient_FDM.py. Run the main file from termianl: main(json_file, finger_count, duty_cycle, current_per_fin). Example command: main(layers.json, 64, 0.5, 50*10**(-6))
# Transient Temperature Analysis in FinFET

## Overview

This repository contains the implementation accompanying the paper:

**N. Karmokar, M. Madhusudan, R. Harjani, and Sachin S. Sapatnekar,**
"Transient Temperature Analysis in FinFET Circuits", ACM/IEEE International Symposium on Low Power Electronics and Design (ISLPED), 2024.

The framework enables transient thermal analysis of FinFET-based circuits by modeling the temporal evolution of temperature under dynamic power dissipation. The goal is to efficiently estimate temperature variations and thermal hotspots without relying exclusively on computationally expensive full-device simulations.

---

## Repository Structure

Transient-Temperature-Analysis-in-FinFET/

  Main.py                     # Main execution script
  
  Temperature_Analysis.py     # Thermal analysis engine
  
  Thermal_Model.py            # Thermal modeling utilities
  
  Input                       # Example input data
  
  Output                      # Generated results
  
  README.md


## Requirements

```bash
pip install numpy scipy matplotlib pandas
```

---

## Running the Tool

```bash
python FDM_Main.py
```

Run the main file from termianl: main(json_file, finger_count, duty_cycle, current_per_fin). 

### Example Command

```bash
main(layers.json, 64, 0.5, 50*10**(-6))
```

The tool reads the circuit and power information, performs transient thermal analysis, and generates temperature profiles over time.

---

## Inputs

### Circuit Description

Input data describing the FinFET circuit structure and thermal network.

### Power Profile

Time-varying power dissipation values used to drive the transient temperature simulation.

### Technology Parameters

Technology-dependent thermal parameters, including:

* Thermal resistance
* Thermal capacitance
* Device geometry information
* Material properties

### Simulation Parameters

* Simulation duration
* Time step
* Initial temperature
* Ambient temperature

---

## Methodology

The analysis flow consists of:

1. Thermal-network construction
2. Power trace processing
3. Transient temperature computation
4. Hotspot identification
5. Temperature visualization

The framework captures the temporal evolution of device temperatures under dynamic operating conditions and identifies thermal hotspots that may affect reliability and performance.

---

## Outputs

### Temperature Profiles

* Temperature versus time
* Peak temperature
* Average temperature

### Thermal Maps

* Spatial temperature distribution
* Hotspot locations

### Analysis Metrics

* Maximum temperature
* Temperature rise
* Thermal time constants

### Visualization

* Temperature waveforms
* Thermal heatmaps
* Summary plots

---

## Citation
If you use this code in academic work, please cite:

N. Karmokar, M. Madhusudan, R. Harjani, and Sachin S. Sapatnekar, "Transient Temperature Analysis in FinFET Circuits,"Proceedings of the ACM/IEEE International Symposium on Low Power Electronics and Design (ISLPED), 2024.

