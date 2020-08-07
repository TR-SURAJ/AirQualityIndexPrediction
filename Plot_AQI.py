# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

def avgdata(aqiyear):

    average = []
    for rows in pd.read_csv("Data/AQI/{}".format(aqiyear),encoding='utf-8',chunksize = 24):
        add_var = 0
        avg = 0.0
        data = []
        df = pd.DataFrame(data = rows)
        for index,row in df.iterrows():
            data.append(row['PM2.5'])
        for i in data:
            if type(i) is float or type(i) is int:
                add_var = add_var + i
                
            elif type(i) is str:
                if i!='NoData' and i!='PwrFail' and i!='---' and i!='InVld':
                    temp = float(i)
                    add_var = add_var + temp
        
        avg = add_var/24
        average.append(avg)
        
    return average


if __name__=="__main__":
    lst2013=avgdata('aqi2013.csv')
    lst2014=avgdata('aqi2014.csv')
    lst2015=avgdata('aqi2015.csv')
    lst2016=avgdata('aqi2016.csv')
    lst2017=avgdata('aqi2017.csv')
    lst2018=avgdata('aqi2018.csv')
    plt.plot(range(0,365),lst2013,label="2013 data")
    plt.plot(range(0,364),lst2014,label="2014 data")
    plt.plot(range(0,365),lst2015,label="2015 data")
    plt.xlabel('Day')
    plt.ylabel('PM 2.5')
    plt.legend(loc='upper right')
    plt.show()
        
        