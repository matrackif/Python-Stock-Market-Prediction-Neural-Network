import argparse
import csv
import datetime
from pandas import read_csv


def from_timestamp_to_unix_time(timestamp: str) -> int:
    # Convert date of form year-month-day to unix time
    timestamp_as_list = timestamp.split('-')
    year = int(timestamp_as_list[0])
    month = int(timestamp_as_list[1])
    day = int(timestamp_as_list[2])
    date = datetime.datetime(year=year, month=month, day=day, tzinfo=datetime.timezone.utc)  # Convert to UTC time
    time = int(date.timestamp())
    # print('Year: ' + str(year) + ' Month: ' + str(month) + ' Day: '
    #       + str(day) + ' Unix Time UTC: ' + str(time))
    return time


def from_unix_time_to_timestamp(time: int) -> str:
    return str(datetime.datetime.utcfromtimestamp(time))


def date_parse(x):
    print('x is: ' + str(x))
    return datetime.datetime.strptime(x, '%Y-%m-%d')


class CSVParser:
    def __init__(self, input_file_path: str, input_mode: str):
        self.mode = input_mode
        self.file_path = input_file_path

    def parse_and_create(self):
        if self.mode == 'regular':
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
                unix_time = from_timestamp_to_unix_time(row[0])
                # print(from_unix_time_to_timestamp(unix_time))
                row[0] = str(unix_time)
                # row.remove(row[len(row) - 1])  # Remove volume
                writer.writerow(row)

        elif self.mode == 'lstm':
            # Parse using pandas instead of csv lib
            dataset = read_csv(self.file_path, parse_dates=['timestamp'], index_col=0,
                               date_parser=date_parse)
            # manually specify column names
            dataset.index.name = 'date'
            # summarize first 5 rows
            print(dataset.head(5))
            # save to file
            dataset.to_csv('../data/lstm_dates.csv')


if __name__ == '__main__':
    print('Converting first column of csv file to unix time')
    parser = argparse.ArgumentParser()
    # The argument is of the form -f or --file.
    # If -f or --file is given... for ex: "main.py -f" but no file is given then the "const" argument specifies the file
    # If no -f or --file option is given at all then the "default" argument specifies the file
    parser.add_argument('-f', '--file', nargs='?', type=str, default='../data/daily_MSFT.csv',
                        const='../data/daily_MSFT.csv', help='Path to input CSV file to be read')
    parser.add_argument('-m', '--mode', nargs='?', type=str, default='regular',
                        const='regular', help='Mode of CSV parser, can be either "regular" or "lstm')
    program_args = vars(parser.parse_args())
    file_path = program_args['file']
    mode = program_args['mode']
    csv_parser = CSVParser(file_path, mode)
    csv_parser.parse_and_create()
