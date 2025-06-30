# # Read space-separated .txt file and write it as comma-separated .csv
# with open("testingData.txt", "r") as infile, open("output.csv", "w") as outfile:
#     for line in infile:
#         # Split on whitespace and re-join with commas
#         csv_line = ",".join(line.strip().split())
#         outfile.write(csv_line + "\n")


# import csv

# # Hour blocks for ap/Kp readings (in 3-hour increments)
# three_hour_blocks = [0, 3, 6, 9, 12, 15, 18, 21]

# with open('geomagnetic_storm_dataset.csv', 'r') as infile, open('full_data.csv', 'w', newline='') as outfile:
#     reader = csv.DictReader(infile)
#     fieldnames = [
#         'YYYY', 'MM', 'DD', 'days', 'days_m', 'Bsr', 'dB',
#         'Kp', 'ap', 'index', 'hour', 'Ap', 'SN', 'F10.7obs', 'F10.7adj', 'D'
#     ]
#     writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#     writer.writeheader()

#     for row in reader:
#         for i in range(8):
#             writer.writerow({
#                 'YYYY': row['YYYY'],
#                 'MM': row['MM'],
#                 'DD': row['DD'],
#                 'days': row['days'],
#                 'days_m': row['days_m'],
#                 'Bsr': row['Bsr'],
#                 'dB': row['dB'],
#                 'Kp': row[f'Kp{i+1}'],
#                 'ap': row[f'ap{i+1}'],
#                 'index': i + 1,
#                 'hour': three_hour_blocks[i],
#                 'Ap': row['Ap'],
#                 'SN': row['SN'],
#                 'F10.7obs': row['F10.7obs'],
#                 'F10.7adj': row['F10.7adj'],
#                 'D': row['D']
#             })


import csv

input_file = 'training_data.csv'
output_file = 'training_data_edited.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)

    # Add the new field to existing headers
    fieldnames = reader.fieldnames + ['storm_occurred']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        try:
            kp_value = float(row['Kp'])
            storm = 1 if kp_value >= 5 else 0
        except ValueError:
            storm = ''  # Leave blank if Kp is not a number

        row['storm_occurred'] = storm
        writer.writerow(row)
