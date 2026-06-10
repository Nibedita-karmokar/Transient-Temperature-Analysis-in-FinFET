In the Project_codes folder there are 2 python files (1) FDM_Main.py, and (2) Transient_FDM.py. Run the main file from termianl: main(json_file, finger_count, duty_cycle, current_per_fin). Example command: main(layers.json, 64, 0.5, 50*10**(-6))

# Transient Temperature Analysis in FinFET

## Overview

This repository contains a finite-difference-based framework for transient thermal analysis of FinFET circuits.

The implementation models heat generation and propagation through FinFET devices and surrounding materials using a 3D thermal RC network derived from the physical layout and technology stack. The framework enables efficient estimation of transient temperature profiles under dynamic operating conditions without requiring computationally expensive full-device simulations.

This work accompanies:
**Nibedita Karmokar, Sai-Wang Tam, Thanh Viet Dinh, Vidya A. Chhabria, Ramesh Harjani, and Sachin S. Sapatnekar,** *“Analyzing the Impact of FinFET Self-Heating on the Performance of RF Power Amplifiers”*, in Proc. International Conference on Computer-Aided Design, IEEE, 2024

---

## Repository Structure

```text
Transient-Temperature-Analysis-in-FinFET/

├── Codes_files/
│   ├── Source_codes/
│   │   ├── FDM_Main.py          # Main execution script
│   │   └── Transient_FDM.py     # Thermal solver
│   │
│   └── layers.json             # Technology description
│
└── README.md
```

---

## Requirements

```bash
pip install numpy scipy matplotlib
```

---

## Running the Tool

Launch Python and execute:

```python
main(layers.json, finger_count, duty_cycle, current_per_fin)
```

Example:

```python
main(layers.json, 64, 0.5, 50e-6)
```

where:

* `finger_count` = number of transistor fingers
* `duty_cycle` = switching activity factor
* `current_per_fin` = current flowing through each fin (A)

---

## Inputs

### Technology Description (`layers.json`)

Technology-specific parameters including:

* Metal layers
* Via structures
* Layer dimensions
* Material properties
* Thermal conductivity
* Density
* Heat capacity

### Device Parameters

Specified in `FDM_Main.py`:

* Fin count
* Fin pitch
* Gate dimensions
* Supply voltage
* Current per fin
* Device geometry

### Simulation Parameters

* Simulation duration
* Time step
* ON/OFF switching intervals
* Ambient temperature

---

## Methodology

The analysis flow consists of:

1. Construction of the FinFET geometry and technology stack
2. Discretization of the structure using a 3D finite-difference grid
3. Extraction of thermal resistance and capacitance values
4. Assembly of a sparse thermal RC network
5. Transient temperature computation over time
6. Visualization of temperature evolution

The thermal model captures heat flow through:

* Silicon substrate
* Buried oxide (BOX)
* Fin structures
* Gate structures
* Metal interconnect layers
* Contact and via structures

---

## Outputs

The framework generates:

### Temperature Profiles

* Temperature versus time
* Peak temperature
* Average temperature

### Thermal Metrics

* Maximum temperature rise
* Thermal transient response
* Heat dissipation characteristics

### Visualization

* Temperature evolution plots
* Spatial temperature distribution
* Thermal hotspot identification

---

## Main Source Files

### FDM_Main.py

Top-level driver script responsible for:

* Technology initialization
* FinFET geometry generation
* Power specification
* Simulation setup

### Transient_FDM.py

Thermal analysis engine responsible for:

* Finite-difference discretization
* Thermal RC network construction
* Sparse matrix assembly
* Transient temperature solution

---

## Citation

If you use this software in academic work, please cite:

```text
**Nibedita Karmokar, Sai-Wang Tam, Thanh Viet Dinh, Vidya A. Chhabria, Ramesh Harjani, and Sachin S. Sapatnekar,** *“Analyzing the Impact of FinFET Self-Heating on the Performance of RF Power Amplifiers”*, in Proc. International Conference on Computer-Aided Design, IEEE, 2024
```
