# # to list all executable files available in the system
# #!/bin/bash
# # Directory to search for executable files
# SEARCH_DIR="$HOME"
# # to list all executable files in the directory
# for file in "$SEARCH_DIR"/*
# do
# 	if [ -x "$file" ] && [ -f "$file" ]
# 	then
#         echo "$file"
# 	fi
# done



# #!/bin/bash
# FILE='users.csv'
# # Read the file line by line
# while IFS= read -r line; do
#         userid=$(echo "$line" | cut -d',' -f1)
# 	rest=$(echo "$line" | cut -d ',' -f2)
# 	username=$(echo "$rest" | cut -d':' -f1)
# 	department=$(echo "$rest" | cut -d':' -f2)
# 	# Print the extracted fields
# 	echo "UserID: $userid"
# 	echo "Username: $username"
# 	echo "Department: $department"
# 	echo "   "
# 	done < "$FILE"


# #awk -F[,:] '{ printf "UserID:%s\nUserName:%s\nUserDepartment:%s\n\n",$1,$2,$3 }' users.csv



# #!/bin/bash
# ls *.fasta > fasta.txt
# MOTIF1="YVDRHPDDTINDYLNSI"
# MOTIF2="MGNHTWDHPDIFEILTTK"
# # File containing the list of files to search
# FIle="fasta.txt"
# while IFS= read -r FILE; do
# if [[ -f "$FILE" ]]; then
# grep -ob "$MOTIF1" "$FILE" | while IFS= read -r line; do
# echo "Motif '$MOTIF1' found in file '$FILE' at position $(echo "$line" | cut -d':' -f1)"
# done
# grep -ob "$MOTIF2" "$FILE" | while IFS= read -r line; do
# echo "Motif '$MOTIF2' found in file '$FILE' at position $(echo "$line" | cut -d':' -f1)"
# done
# else
# echo "File '$FILE' does not exist."
# fi
# done < "$fasta.txt"

# #!/bin/bash
# threshold=70
# # to get the percentage of used disk space
# used=$(df "$HOME" | awk 'NR==2 {print $5}' | sed 's/%//')
# # Check if the usage percentage crosses the threshold
# if [ "$used" -ge "$threshold" ]; then
#     echo "This exceeds the threshold of $threshold %."
# else
# 	echo "This doesn't exceed the threshold of $threshold %."
# fi


# #!/bin/bash
# root_total=$(df / | awk 'NR==2 {print $2}')
# home_used=$(df /home | awk 'NR==2 {print $3}')
# # to calculate percentage of home usage relative to root
# percent=$(( ( home_used * 100) / root_total ))
# echo "Total disk space of root: $root_total KB"
# echo "Used disk space of home: $home_used KB"
# echo "Percentage of home directory usage relative to root: $percent%"


# #!/bin/bash
# pdir="/home/ibab" 
# # Check if the directory exists
# if [ ! -d "$pdir" ]; then
#     echo "Error: Directory '$pdir' not found."
#     exit 1
# fi
# # Find and check each subdirectory
# for dir in "$pdir"/*/; do
#     # Check if the item is a directory and is empty
#     if [ -d "$dir" ] && [ -z "$(ls -A "$dir")" ]; then
#         echo "$dir" >> op.txt
#     fi
# done
# echo "List of empty subdirectories saved."






# #!/bin/bash
# in="c.txt"
# # to check if the input file exists
# if [ ! -f "$in" ]; then
#     echo "Error: File '$in' not found."
#     exit 1
# fi
# # to remove duplicate lines and overwrite the original file
# sort "$in" | uniq >> "$in"
# cat "$in"



# #!/bin/bash
# # to set the threshold value
# threshold=90
# # Looping through numbers 1 to 100
# for (( i=1; i<=100; i++ ))
# do
#         echo "$i"
#     	# Check if the number is greater than the threshold
#     if [ "$i" -gt "$threshold" ]; then
#         echo "$i is greater than $threshold"
#     fi
# done



# #!/bin/bash
# # To get the current hour using the `date` command
# time=$(date +%H)

# # Determine the appropriate greeting based on the current hour
# if [ "$time" -ge 5 ] && [ "$time" -lt 12 ]; then
#     echo "Good morning!"
# elif [ "$time" -ge 12 ] && [ "$time" -lt 18 ]; then
#     echo "Good afternoon!"
# else
#     echo "Good night!"
# fi


#!/bin/bash
DNA_STRING="ACGTACGGTACG"
#Initialize counters for each nucleotide
A=0;C=0;G=0;T=0
# Looping through each character in the DNA string
for nucleotide in $(echo "$DNA_STRING" | fold -w1)
do
    if [[ "$nucleotide" == "A" ]]; then
        A=$((A + 1))
    elif [[ "$nucleotide" == "C" ]]; then
        C=$((C + 1))
    elif [[ "$nucleotide" == "G" ]]; then
        G=$((G + 1))
    elif [[ "$nucleotide" == "T" ]]; then
        T=$((T + 1))
    else
        echo "Error: DNA string contains invalid characters."
        exit 1
    fi
done

# Print the results
echo "Count of A: $A"
echo "Count of C: $C"
echo "Count of G: $G"
echo "Count of T: $T"








