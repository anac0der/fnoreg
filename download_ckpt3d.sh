#!/bin/bash

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown could not be found, please install it (e.g., pip install gdown)"
    exit 1
fi

# URLs and corresponding IDs
urls=(
    "https://drive.google.com/file/d/1AHoUe0txeG0kwQP2Sn6IplCm8SIUfLCr/view?usp=sharing"
    "https://drive.google.com/file/d/1rVVTzH0kG97UBb6xY025PCU6pIvA8udQ/view?usp=sharing"
    "https://drive.google.com/file/d/1yJ-ojT48S1GEXpbLHZR-MLtlxSBNKMIr/view?usp=sharing"
    "https://drive.google.com/file/d/1poE0w2fqJ_Wy8-1m0R_KT2MgmZ8ns4ve/view?usp=sharing"
    "https://drive.google.com/file/d/1fIyyAGlEkUafHqaMouSDy7AFu0dmACGa/view?usp=sharing"
    "https://drive.google.com/file/d/1Yty5gjk8XIMkZVMyvrZq-1IsEXyJYOoC/view?usp=sharing"
    "https://drive.google.com/file/d/1EsVLuvBHNRLJVfW2fUMmUs50xE5zuApB/view?usp=sharing"
    "https://drive.google.com/drive/folders/159SEpbIpd1uo2C8_n3icAA8rJj4x-KNu?usp=sharing"
)

ids=(
   "23"
   "27"
   "29"
   "31"
   "32"
   "35"
   "38"
   "39"
)

urls_vxm=(
    "https://drive.google.com/file/d/1ViWTj2thTXscHRnFaCynhG_Hq7lpZIbi/view?usp=sharing"
    "https://drive.google.com/file/d/1e1bv9_B6eKIMzbBDk-VbZ6cUI5miscLv/view?usp=sharing"
    "https://drive.google.com/file/d/1e7T4Y25ZS63TfhMP2L2uQEMLAcVp3qUQ/view?usp=sharing"
    "https://drive.google.com/file/d/16lS3mMY-0C5JxREJJ3SyzdCIl4hDmh8x/view?usp=sharing"
    "https://drive.google.com/drive/folders/1FRkjCGt01ZseRaOKX9Y1yov5ZCU6wu4r?usp=sharing"
)

ids_vxm=(
    "0"
    "2"
    "3"
    "5"
    "6"
)

# Output base directory
output_base_dir="experiments_fourier"
output_base_dir_vxm="baseline_models/transmorph/experiments"

# Function to download files
download_checkpoint() {
    local url="$1"
    local exp_id="$2"
    local checkpoint_dir="$3"
    local filename="$4"

    if [ -f "$checkpoint_dir/$filename" ]; then
        echo "File $filename already exists in $checkpoint_dir. Skipping download."
        return
    fi

    echo "Downloading checkpoint $exp_id to $checkpoint_dir/$filename"
    gdown "$url" -O "$checkpoint_dir/$filename" --fuzzy

    if [ $? -eq 0 ]; then
        echo "Checkpoint $exp_id downloaded successfully."
    else
        echo "Failed to download checkpoint $exp_id."
    fi
}

# Loop through the URLs and IDs
for i in "${!urls[@]}"; do
    exp_id="${ids[$i]}"
    exp_id_int=$((exp_id))

    # Determine filename based on ID
    checkpoint_filename="weights.pth"

    # Create the directory for the experiment
    checkpoint_dir="$output_base_dir/oasis_v_exp${exp_id}"
    mkdir -p "$checkpoint_dir"

    # Call download function
    download_checkpoint "${urls[$i]}" "$exp_id" "$checkpoint_dir" "$checkpoint_filename"
done

for i in "${!urls_vxm[@]}"; do
    exp_id="${ids_vxm[$i]}"

    # Determine filename based on ID
    checkpoint_filename="weights.pth"

    # Create the directory for the experiment
    checkpoint_dir="$output_base_dir_vxm/oasis_v_exp${exp_id}"
    mkdir -p "$checkpoint_dir"

    # Call download function
    download_checkpoint "${urls[$i]}" "$exp_id" "$checkpoint_dir" "$checkpoint_filename"
done

echo "All downloads completed."