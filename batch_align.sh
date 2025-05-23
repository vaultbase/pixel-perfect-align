#!/bin/bash
# Batch alignment script - just drag folders onto this

echo "🚀 Pixel Perfect Align - Batch Mode"
echo ""

if [ $# -eq 0 ]; then
    echo "Usage: Drop folders onto this script"
    echo "Or run: ./batch_align.sh /path/to/folder1 /path/to/folder2 ..."
    exit 1
fi

# Process each folder
for folder in "$@"; do
    if [ -d "$folder" ]; then
        echo "Processing: $folder"
        python3 simple_align.py "$folder"
        echo "---"
    fi
done

echo "✅ Batch processing complete!"