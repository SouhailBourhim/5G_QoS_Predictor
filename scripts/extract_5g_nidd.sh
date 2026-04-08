#!/bin/bash

# 5G-NIDD Dataset Extraction Script

NIDD_DIR="data/raw/5g_nidd"

echo "🔓 Extracting 5G-NIDD Dataset Files..."
echo "========================================"
echo ""

cd "$NIDD_DIR" || exit 1

# Extract the most relevant files for QoS analysis
echo "📦 Extracting Combined dataset..."
unzip -q "Combined.zip" -d "combined"

echo "📦 Extracting BTS1_BTS2 fields preserved..."
unzip -q "BTS1_BTS2_fields_preserved.zip" -d "bts_combined"

echo "📦 Extracting BS1 CSV attacks (for feature reference)..."
unzip -q "BS1_each_attack_csv.zip" -d "bs1_attacks"

echo "📦 Extracting BS2 CSV attacks (for feature reference)..."
unzip -q "BS2_each_attack_csv.zip" -d "bs2_attacks"

echo ""
echo "✅ Extraction complete!"
echo ""
echo "📁 Extracted directories:"
ls -d */ | grep -v "\.zip"
