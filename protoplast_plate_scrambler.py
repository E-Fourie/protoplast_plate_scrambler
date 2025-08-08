import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random, argparse, collections
import matplotlib.colors as mcolors

def create_plate():
    """Create a 96-well plate layout as a dictionary."""
    rows = 'ABCDEFGH'
    cols = [str(i) for i in range(1, 13)]
    return {row: [f"{row}{col}" for col in cols] for row in rows}

def flatten_plate(plate):
    """Flatten the plate dict into a list by row."""
    return [well for row in plate.values() for well in row]

def replace_well(plate, well_name, value):
    """Replace the value in a specific well of the plate."""
    row = well_name[0]
    col_idx = int(well_name[1:]) - 1
    plate[row][col_idx] = value

def generate_sample_list(num_samples, num_replicates):
    """Generate a list of sample numbers with replicates."""
    sample_list = []
    for sample_num in range(1, num_samples + 1):
        sample_list.extend([sample_num] * num_replicates)
    random.seed(1)
    random.shuffle(sample_list)
    return sample_list

def get_psuedoY(x):
    """Convert x index to pseudo Y value for 96-well plate."""
    mapping = {
        0: '7', 1: '6', 2: '5', 3: '4', 4: '3', 5: '2', 6: '1',
        -1: '8', -2: '9', -3: '10', -4: '11', -5: '12'
    }
    return mapping.get(x, None)


def get_line_by_y(y):
    """Convert y index to plate row letter."""
    # Mapping y index to row letter
    y_to_row = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}
    return y_to_row.get(y, None)

def fill_plate(samples, plate, exclude_rows=None):
    """
    Fill the plate with sample numbers based on sample list.
    exclude_rows: list of rows (0-based indices) to exclude (assign zero).
    """
    exclude_rows = exclude_rows or []
    sample_iter = iter(samples)

    x = y = 0
    while y <= 7:
        if y in exclude_rows:
            value = 0
        else:
            try:
                value = next(sample_iter)
            except StopIteration:
                value = 0

        psuedoY = get_psuedoY(x)
        psuedoX = get_line_by_y(y)
        if psuedoY and psuedoX:
            well_name = f"{psuedoX}{psuedoY}"
            replace_well(plate, well_name, value)

        y += 1
        if y > 7:
            y = 0
            # Adjust x for zig-zag pattern
            if x >= 6:
                break
            if x > 0:
                x = -x
            else:
                x = -x + 1

def remainder_spiral(samples, plate):
    """
    Fill the plate in a spiral pattern dynamically based on the plate size.
    """

    sample_iter = iter(samples)

    rows = list(plate.keys())          # e.g. ['A','B','C','D','E','F','G','H']
    cols = list(range(1, len(plate[rows[0]]) + 1))  # e.g. [1..12]

    max_y = len(rows)      # number of rows, e.g. 8
    max_x = len(cols)      # number of columns, e.g. 12

    # Spiral bounds: from 0 to max_x -1, 0 to max_y -1
    visited = set()
    x = y = 0
    dx, dy = 1, 0  # start moving right

    for _ in range(max_x * max_y):
        if 0 <= x < max_x and 0 <= y < max_y and (x, y) not in visited:
            try:
                value = next(sample_iter)
            except StopIteration:
                value = 0

            row_letter = rows[y]
            col_number = cols[x]
            well_name = f"{row_letter}{col_number}"
            replace_well(plate, well_name, value)
            visited.add((x, y))

        # Calculate next position
        nx, ny = x + dx, y + dy

        # Change direction if next pos out of bounds or already visited
        if nx < 0 or nx >= max_x or ny < 0 or ny >= max_y or (nx, ny) in visited:
            dx, dy = -dy, dx  # turn right: (dx,dy) = (-dy, dx)
            nx, ny = x + dx, y + dy

        x, y = nx, ny

def descrambler(inputFile, plate):
    """Descramble a plate of samples from a CSV or Excel file."""
    try:
        data = pd.read_csv(inputFile) 
    except:
        data = pd.read_excel(inputFile)
    dfIncoming = pd.DataFrame(data, columns=["1","2","3","4","5","6","7","8","9","10","11","12"])
    dfIncoming = dfIncoming.fillna(0)

    valueDict = collections.defaultdict(list)

    row_labels = list('ABCDEFGH')
    for row_label in row_labels:
        plate_row = plate[row_label]  # Sample numbers for the row
        data_row = dfIncoming.loc[row_labels.index(row_label)].tolist()  # Corresponding data from input

        for sample_num, data_val in zip(plate_row, data_row):
            if sample_num != 0:  # skip excluded wells
                valueDict[sample_num].append(data_val)

    # Sort by sample number
    valueDict = collections.OrderedDict(sorted(valueDict.items()))

    exportdf = pd.DataFrame.from_dict(valueDict, orient='index')
    exportdf.columns = [f"replicate_{i+1}" for i in range(exportdf.shape[1])]
    exportdf = exportdf.reset_index().rename(columns={'index': 'ID'})
    replicate_cols = exportdf.columns.difference(['ID'])
    exportdf['average'] = exportdf[replicate_cols].mean(axis=1)
    exportdf['SE'] = exportdf[replicate_cols].std(axis=1, ddof=1) / np.sqrt(len(replicate_cols))
    exportdf.to_csv("outputData.csv", index=False)

def plate_output(plate):
    """Visualize the plate layout using matplotlib."""
    df_plate = pd.DataFrame.from_dict(plate, orient='index')

    _, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(df_plate.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int), cmap=mcolors.ListedColormap(['white'] + [plt.cm.tab20(i) for i in range(plt.cm.tab20.N)]), interpolation='none')
    ax.set(xticks=np.arange(12), yticks=np.arange(8), xticklabels=range(1,13), yticklabels=list('ABCDEFGH'))
    ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-0.5,12,1), minor=True); ax.set_yticks(np.arange(-0.5,8,1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1); ax.grid(which='major', visible=False)
    for i in range(df_plate.shape[0]):
        for j in range(df_plate.shape[1]):
            val = df_plate.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).iat[i,j]
            if val != 0:
                ax.text(j, i, str(val), ha='center', va='center')
    plt.title('96-Well Plate Layout\n')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="96-well Plate Scramble")
    parser.add_argument("Samples", metavar="S", type=int, help="Number of samples")
    parser.add_argument("Replicates", metavar="R", type=int, help="Number of replicates")
    parser.add_argument("--descramble", "-ds", type=str, help="Descramble a plate of samples. Usage -ds [file.csv]")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    sample_number = args.Samples
    replicate_number = args.Replicates 
    descramble = args.descramble
    plate = create_plate() 
    sample_list = generate_sample_list(sample_number, replicate_number)

    exclusion_map = {
        8: [],
        7: [0],
        6: [0, 7],
        5: [0, 1, 7],
        4: [0, 1, 6, 7],
    }
    total = sample_number * replicate_number
    exclusion_map_param = None
    max_limits = {4: 48, 5: 60, 6: 72, 7: 84, 8: 96}

    for n in [4, 5, 6, 7, 8]:
        if total % n == 0 and total <= max_limits[n]:
            exclusion_map_param = n
            break

    if exclusion_map_param is not None:
        fill_plate(sample_list, plate, exclude_rows=exclusion_map[exclusion_map_param])
    else:
        remainder_spiral(sample_list, plate)

    if descramble:
        descrambler(descramble, plate)
    else:
        plate_output(plate)

if __name__ == "__main__":
    main()
