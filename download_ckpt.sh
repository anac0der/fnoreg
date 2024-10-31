#!/bin/bash

# Ensure gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown could not be found, please install it (e.g., pip install gdown)"
    exit 1
fi

# URLs and corresponding IDs
urls=(
    "https://drive.google.com/file/d/1wtJic1qOcIkQmVvYK8d4-U0XJko8mLW1/view?usp=sharing"
    "https://drive.google.com/file/d/153BQBjxLIDnwni1-VgotvLEFEJkxfftN/view?usp=sharing"
    "https://drive.google.com/file/d/1RkgUqaAWvouKN4f9LCaGh_pxyVD2boym/view?usp=sharing"
    "https://drive.google.com/file/d/1eu_rXs2XAh1sc-TELnLgqsnC15ECQ62Q/view?usp=sharing"
    "https://drive.google.com/file/d/1vbjlN7x7nx4CEAXAZTOxXYNFCRz-7J2k/view?usp=sharing"
    "https://drive.google.com/file/d/13Ooz4e9ka2FwageqYbbKkU8hC1H3VJOu/view?usp=sharing"
    "https://drive.google.com/file/d/1e55wkuuiqdl9sq6dj-Ndwk_e08ynPj0N/view?usp=sharing"
    "https://drive.google.com/file/d/1NTyLo1uUtKSRBwlUhmZ70jxKZTO3pRsA/view?usp=sharing"
    "https://drive.google.com/file/d/1XJobYT_JrIxORcpzCz_ILRlaoXZGG_EP/view?usp=sharing"
    "https://drive.google.com/file/d/14kYpuQwB4voDE3TST5ptPdyyelfGtIn6/view?usp=sharing"
    "https://drive.google.com/file/d/1siWLVlZHt6L7nFjvOxLQHDlPiGjjOCmD/view?usp=sharing"
    "https://drive.google.com/file/d/1EgEfKKVm3Jka5Cx9EwvaZZ7uShIarUBw/view?usp=sharing"
    "https://drive.google.com/file/d/1mvmL8G8yYJQ9YuAwCDX4t7Ix0iNTn_gJ/view?usp=sharing"
    "https://drive.google.com/file/d/1k1OmumbPZ7EbipEmQGF90DguhIx96a3F/view?usp=sharing"
    "https://drive.google.com/file/d/1c5dYpnUPMP6mDJCFYm6eY192XtvODyUU/view?usp=sharing"
    "https://drive.google.com/file/d/1NHJpa1WTvwkXk68KGJWV3AQwB5YfV4Be/view?usp=sharing"
    "https://drive.google.com/file/d/1ar1WZcLYaDFKF4NkDlN9u2rQHPnTnDqN/view?usp=sharing"
    "https://drive.google.com/file/d/1ocnm-c45UH4IG-NH2Ygl-7vyd79HXLDt/view?usp=sharing"
)

ids=(
    "54"
    "65"
    "69"
    "70"
    "71"
    "74"
    "75"
    "76"
    "77"
    "78"
    "79"
    "80"
    "81"
    "82"
    "83"
    "84"
    "139"
    "140"
)

urls_vxm=(
   "https://drive.google.com/file/d/1nmJoqV0p6glnQ0yAhXJuZ5SdJ5gNRgYn/view?usp=sharing"
   "https://drive.google.com/file/d/1g9fYhppsvdJSoxZs2k8i1Zux0twpOVFO/view?usp=sharing"
   "https://drive.google.com/file/d/1MKJXuBnqJ9hmdMxk0myCN5vaIp2u9B84/view?usp=sharing"
   "https://drive.google.com/file/d/1xaroxhsd3_ffY3p-NcgHf4S5mpdobJiT/view?usp=sharing"
   "https://drive.google.com/file/d/18IpGq4FKHLQfxGuq2WIitLR8UGpmFWus/view?usp=sharing"
)

ids_vxm=(
    "1"
    "2"
    "13"
    "27"
    "28"
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
    if [ "$exp_id_int" -eq 78 ] || [ "$exp_id_int" -eq 82 ] || [ "$exp_id_int" -eq 139 ]; then
        checkpoint_filename="weights.pt"
    else
        checkpoint_filename="weights.pth"
    fi

    # Create the directory for the experiment
    checkpoint_dir="$output_base_dir/oasis_exp${exp_id}"
    mkdir -p "$checkpoint_dir"

    # Call download function
    download_checkpoint "${urls[$i]}" "$exp_id" "$checkpoint_dir" "$checkpoint_filename"
done

for i in "${!urls_vxm[@]}"; do
    exp_id="${ids_vxm[$i]}"

    # Determine filename based on ID
    checkpoint_filename="weights.pth"

    # Create the directory for the experiment
    checkpoint_dir="$output_base_dir_vxm/oasis_exp${exp_id}"
    mkdir -p "$checkpoint_dir"

    # Call download function
    download_checkpoint "${urls[$i]}" "$exp_id" "$checkpoint_dir" "$checkpoint_filename"
done

echo "All downloads completed."