import numpy as np

def make_mpg_data(filename='auto-mpg.data'):

    def missingIsNan(s):
        return np.nan if s == b'?' else float(s)

    data = np.loadtxt(filename, usecols=range(8), converters={3: missingIsNan})

    print("Read",data.shape[0],"rows and",data.shape[1],"columns from",filename)

    goodRowsMask = np.isnan(data).sum(axis=1) == 0
    data = data[goodRowsMask,:]

    print("After removing rows containing question marks, data has",data.shape[0],"rows and",data.shape[1],"columns.")
    x = data[:,1:]
    t = data[:,0:1]

    x_names =  ['bias', 'cylinders','displacement','horsepower','weight','acceleration','year','origin']
    t_names = 'mpg'
    return x, t, x_names, t_names

def read_appliances_Data(filepath):
    dataWithoutHeader = pd.read_csv(filepath, usecols=range(1,27)).as_matrix()

    x_names = ['T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']

    t_names = ['Appliances','lights']

    x = np.asarray(dataWithoutHeader[:,2:])

    t = np.asarray(dataWithoutHeader[:,:2])

    return x, t, x_names, t_names
