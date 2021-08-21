# ventparliament
Benchmarking Suite/Data for Single Chamber Compliance/Resistance Modeling Algorithms.

## Citing

If you make use of this in your own publications please cite us

`XXX`

## Install

`ventparliament` can be utilized with python3. Install this for use in your own python projects

    conda env create -f environment.yml
    conda activate parliament

## Usage

### Getting Started

You can quickly use `ventparliament` in your own projects after installation

```python
from parliament.analyze import FileCalculations

calcs = FileCalculations('patient_name', '/path/to/file.csv', ['polynomial'], 5, [])
compliance_calcs = calcs.quick_analyze_file()
```

### Algorithms Supported

 * Al-Rawas method
 * Flow-targeted inspiratory least squares
 * Howe's expiratory least squares
 * Kannangara's Method
 * IIPR
 * IIMIPR
 * IIPREDATOR
 * Major's method (basically max-pooling PREDATOR)
 * MIPR
 * Polynomial Method
 * PREDATOR
 * Pressure-targeted expiratory least squares
 * Pressure-targeted inspiratory least squares
 * Vicario's constrained optimization
 * Vicario's non-invasive estimation of alveolar pressure

### Benchmarking Run

`XXX`

### Utilizing Different Time Constants

`XXX`

## Contributions

Patches/Fixes/Features are greatly appreciated. Just submit a pull and we'll get to them quickly
