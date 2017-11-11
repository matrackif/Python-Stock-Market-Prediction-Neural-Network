import argparse
import csv
import datetime

if __name__ == '__main__':
    print('Converting first column of csv file to unix time')
    parser = argparse.ArgumentParser()
    # The argument is of the form -f or --file.
    # If -f or --file is given... for ex: "main.py -f" but no file is given then the "const" argument specifies the file
    # If no -f or --file option is given at all then the "default" argument specifies the file
    parser.add_argument('-f', '--file', nargs='?', type=str, default='../data/daily_MSFT.csv',
                        const='../data/daily_MSFT.csv', help='Path to input CSV file to be read')
    program_args = vars(parser.parse_args())
    file_path = program_args['file']
    data = None
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)  # Here we store the input file
        # data.remove(data[0])  # Remove header row
        print(data)

    writer = csv.writer(open('../data/corrected_dates.csv', 'w', newline=''), delimiter=',')
    # https://stackoverflow.com/questions/3348460/csv-file-written-with-python-has-blank-lines-between-each-row
    count = 0
    for row in data:
        if count == 0:  # Treat header row differently
            count += 1
            writer.writerow(row)
            continue
        timestamp_as_list = row[0].split('-')
        year = int(timestamp_as_list[0])
        month = int(timestamp_as_list[1])
        day = int(timestamp_as_list[2])
        date = datetime.datetime(year=year, month=month, day=day, tzinfo=datetime.timezone.utc)  # Convert to UTC time
        unix_time = int(date.timestamp())
        print('Year: ' + str(year) + ' Month: ' + str(month) + ' Day: '
              + str(day) + ' Unix Time UTC: ' + str(unix_time))
        # print(datetime.datetime.utcfromtimestamp(unix_time))
        row[0] = str(unix_time)
        # row.remove(row[len(row) - 1])  # Remove volume
        writer.writerow(row)
