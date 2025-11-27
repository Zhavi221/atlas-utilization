#!/bin/bash
# Usage: ./check_script_io.sh <script_to_run>

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_to_run>"
    exit 1
fi

SCRIPT="$1"

# Check if script exists
if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: File not found: $SCRIPT"
    exit 1
fi

# Make sure script is executable
if [[ ! -x "$SCRIPT" ]]; then
    chmod +x "$SCRIPT"
fi

# Check if pidstat exists
if ! command -v pidstat &>/dev/null; then
    echo "Error: pidstat not found. Install sysstat package."
    exit 1
fi

# Run the script in background
./"$SCRIPT" &
PID=$!

echo "Monitoring I/O of PID $PID..."

# Start pidstat to log read/write every second
pidstat -d -p $PID 1 > io_usage.log &
PIDSTAT_PID=$!

# Small delay to ensure pidstat starts
sleep 1

# Wait for script to finish
wait $PID

# Stop pidstat
kill $PIDSTAT_PID 2>/dev/null

# Ensure log file exists
if [[ ! -f io_usage.log ]]; then
    echo "Error: io_usage.log was not created."
    exit 1
fi

echo "Monitoring finished. Per-second I/O logged in io_usage.log."

# Calculate peak read/write in MB/s
PEAK_RD=$(awk 'NR>3 {if($4>max) max=$4} END{print max/1024}' io_usage.log)
PEAK_WR=$(awk 'NR>3 {if($5>max) max=$5} END{print max/1024}' io_usage.log)

# Calculate total read/write in MB
TOTAL_RD=$(awk 'NR>3 {sum+=$4} END{print sum/1024}' io_usage.log)
TOTAL_WR=$(awk 'NR>3 {sum+=$5} END{print sum/1024}' io_usage.log)

echo ""
echo "Peak Read: ${PEAK_RD:-0} MB/s"
echo "Peak Write: ${PEAK_WR:-0} MB/s"
echo "Total Read: ${TOTAL_RD:-0} MB"
echo "Total Write: ${TOTAL_WR:-0} MB"

# Suggest a safe PBS io flag (add 20% buffer to total I/O)
SAFE_IO=$(awk -v rd=$TOTAL_RD -v wr=$TOTAL_WR 'BEGIN{printf "%.0f", (rd+wr)*1.2}')
echo ""
echo "Suggested PBS -l io value (MB): $SAFE_IO"
