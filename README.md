# 96-Well Plate Protoplast Layout Scrambler & Descrambler

Python utilities for arranging and analyzing 96-well plate layouts:

- **Randomize plate layout** for a given number of samples and replicates  
- **Exclude specific rows** based on sample number  
- **Fill in spiral pattern** for leftover samples  
- **Descramble results** from CSV/Excel files into grouped replicates with averages and standard errors  
- **Visualize plate layout** as a color-coded grid  

## Installation

```bash
pip install numpy pandas matplotlib
```

## Usage
```bash
python plate_scrambler.py [Samples] [Replicates] [options]
```

## Options
| Flag    | Description                                            | Example                  |
|---------|--------------------------------------------------------|--------------------------|
| S    | 	Number of samples to place on the plate                            | `8`              |
| R  | Number of replicates per sample          | `4`              |
| `-ds`  | Descramble plate results from CSV/Excel. Outputs outputData.csv with grouped replicates, mean, and SEs         | `-ds result.csv` |
