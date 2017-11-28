from pandas import concat
from pandas import DataFrame
from pandas import to_datetime
import matplotlib
import datetime


# See this link for more details on this function:
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        In other words n_in is how many days before we shall predict, n_out is how many days ahead
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(-i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def convert_to_matplot_dates(dataframe):
    dates_dataframe = dataframe['timestamp'].apply(lambda x: from_timestamp_to_unix_time(x))
    print(str(dates_dataframe.head()))
    datetimes = to_datetime(dates_dataframe.values, unit='s')
    list_datetimes = datetimes.to_pydatetime().tolist()
    return matplotlib.dates.date2num(list_datetimes)


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
