#!/bin/bash

# Define the file pattern
file_pattern="ergebnis_Stefan_2024-06-07_1*"

# Initialize variables to store the maximum values
max_reward=-1
max_cash=-1

# Temporary files to buffer the output
temp_file=$(mktemp)

# Read the files and extract the values, buffering the output
cat $file_pattern | awk '{print $4, $6}' > $temp_file

# Process the buffered output
while read reward cash; do
    # Print the extracted values
    echo "Reward: $reward, Cash: $cash"
    
    # Convert the cash value to float
    cash=$(echo $cash | sed 's/,/./g')
    
    # Update the maximum reward
    if (( $(echo "$reward > $max_reward" | bc -l) )); then
        max_reward=$reward
    fi
    
    # Update the maximum cash
    if (( $(echo "$cash > $max_cash" | bc -l) )); then
        max_cash=$cash
    fi
    
    # Print the current maximum values
    echo "Current Maximum Reward: $max_reward"
    echo "Current Maximum Cash: $max_cash"
    
done < $temp_file

# Clean up temporary file
rm -f $temp_file

# Output the final maximum values
echo "Final Maximum Reward: $max_reward"
echo "Final Maximum Cash: $max_cash"
