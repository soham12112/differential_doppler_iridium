#!/bin/bash
# Decode Iridium frames from captured data (harder mode)

# Update these paths for your experiment
INPUT_FILE="/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/b200_28/output.bits"
OUTPUT_FILE="/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/Data/b200_28/decoded.txt"
PARSER="/Users/sohamdhirendesai/Desktop/Projects_2/PNT/Iridium_analysis/iridium-toolkit/iridium-parser.py"

python3 "$PARSER" -p "$INPUT_FILE" --harder >> "$OUTPUT_FILE"

echo "Decoding complete. Press any key to continue..."
read -n 1 -s

