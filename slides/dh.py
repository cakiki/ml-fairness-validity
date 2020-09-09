#this is a python script to import as extension in ipynb 
#the functions perform transformation on the data used for timeseries analysing 


import sklearn
import pandas as pd
import numpy as np
import holoviews as hv
import re
import matplotlib.pyplot as plt
from holoviews import opts
from bokeh.models import LinearAxis, Range1d, GlyphRenderer
from holoviews.plotting.links import RangeToolLink
from IPython.display import display, HTML

from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics





def dat_facebook_states(df, state):
    df = df[df["polygon_name"]==state]
    df.index = pd.to_datetime(df["ds"]) - timedelta(days=0)
    df = df[df["ds"]< pd.to_datetime('2020-06-27', format='%Y%m%d', errors='ignore')]
    return(df[["all_day_bing_tiles_visited_relative_change","all_day_ratio_single_tile_users"]])

def dat_rki_states(rki, state):
    rki = rki[rki["Bundesland"]==state][["AnzahlFall", "Refdatum"]].groupby(["Refdatum"]).sum()
    rki.index = pd.to_datetime(rki.index) 
    return(rki)

def dat_apple_states(df, state):
    df = df[(df["region"]==state) & (df["transportation_type"]=="driving")].transpose().iloc[6:]
    df.index = pd.to_datetime(df.index)
    df = df[df.index< pd.to_datetime('2020-06-27', format='%Y%m%d', errors='ignore')]
    df.columns = ["driving"]
    return df

def dat_google_states(df, state):
    df.index = pd.to_datetime(df["date"])
    return(df[df["sub_region_1"]==state].loc[:,"retail_and_recreation_percent_change_from_baseline":])


facebook_states = ['Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland',
       'Sachsen-Anhalt', 'Sachsen', 'Schleswig-Holstein', 'Th-ringen',
       'Baden-W-rttemberg', 'Bayern', 'Brandenburg', 'Bremen',
       'Hamburg', 'Hessen', 'Niedersachsen']
#, 'Berlin' , 'Mecklenburg-Vorpommern'

rki_states = ['Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland', 
              'Sachsen-Anhalt', 'Sachsen', 'Schleswig-Holstein',  'Thüringen', 
              'Baden-Württemberg','Bayern', 'Brandenburg', 'Bremen',
              'Hamburg', 'Hessen', 'Niedersachsen']
#, 'Berlin', 'Mecklenburg-Vorpommern'

apple_states = ['North Rhine-Westphalia', 'Rhineland-Palatinate', 'Saarland',
                'Saxony-Anhalt', 'Saxony', 'Schleswig-Holstein', 'Thuringia',
                'Baden-Württemberg', 'Bavaria', 'Brandenburg', 'Bremen (state)', 
                'Hamburg', 'Hesse', 'Lower Saxony']

#, 'Berlin', 'Mecklenburg-Vorpommern'
google_states = ['North Rhine-Westphalia', 'Rhineland-Palatinate', 'Saarland', 
                 'Saxony-Anhalt', 'Saxony', 'Schleswig-Holstein', 'Thuringia',
                 'Baden-Württemberg', 'Bavaria', 'Brandenburg','Bremen', 
                 'Hamburg', 'Hesse', 'Lower Saxony']
#, 'Berlin','Mecklenburg-Vorpommern'


def shift_days(df, days_min=0, days_max=15, drop_cases=True, drop_na=True):
    fin=df       
    for i in range(1+days_min, days_max+1):

        if(drop_cases):
          df = df.drop('#cases', axis=1)

        dfy = df.shift(i)
        dfy.columns = [ s + " t-" + str(i) for s in dfy.columns]
        fin = pd.concat([fin, dfy], axis=1)
        if(drop_na):
          fin = fin.dropna()
    return(fin)


def join_dat(tiles, apple, google, rki):
    df = tiles.join(apple).join(google).join(rki)
    df["AnzahlFall"] = df["AnzahlFall"].fillna(0)
    return df
    


def prepare_data(all_sources=True, 
                 states=True, 
                 drop_cases=False, 
                 drop_na=False, 
                 days_max=10, 
                 days_min=0, 
                 drop_today=True):

    google = pd.read_csv("../data/Global_Mobility_Report.csv")
    apple = pd.read_csv ("../data/applemobilitytrends-2020-06-29.csv")
    facebook = pd.read_csv ("../data/movement-range-2020-08-13.txt", "\t")
    rki = pd.read_csv ("../data/RKI_COVID19.csv")

    dat = []

    for i in range(len(facebook_states)):
        df = join_dat(dat_facebook_states(facebook, facebook_states[i]),
                      dat_apple_states(apple, apple_states[i]),
                      dat_google_states(google, google_states[i]),
                      dat_rki_states(rki, rki_states[i]))

        df.index.name = "date"
        nam = ['bing_tiles_visited',
           'single_tile_users', 'driving',
           'retail_and_recreation',
           'grocery_and_pharmacy',
           'parks',
           'transit_stations',
           'workplaces',
           'residential', '#cases']
        df.columns = nam
        df = shift_days(df, days_min=days_min, days_max=days_max, drop_cases=drop_cases, drop_na=drop_na)
        if(states):
            df["state"]=i
        if(drop_today):
            dat.append(df.iloc[:,9:len(df.columns)])
        else:
            dat.append(df)
        
        
        
    return(dat)



def train_test_data(dat, diffs=False, testsize=0.2):
    
    iv_train = []
    dv_train = []
    iv_test = []
    dv_test = []
    begin = []


    for i in range(len(facebook_states)):

        begin.append(dat[i]["#cases"][0])

        if(diffs):
            dat[i]["#cases"] = dat[i]["#cases"].diff()
            
        dat[i] = dat[i].dropna()
        X_train, X_test, y_train, y_test = train_test_split(dat[i].drop("#cases", axis=1), 
                                                            dat[i]["#cases"], test_size=0.2, shuffle=False)
        test_date_range = y_test.index
        iv_train.append(X_train)
        dv_train.append(y_train)
        iv_test.append(X_test)
        dv_test.append(y_test)
        
    return([iv_train, dv_train, iv_test, dv_test, begin])
    

def train_model(iv_train, iv_test, dv_train, n_ests=10):

    y_train = pd.concat(dv_train)
    
    scaler_features = StandardScaler()
    scaler_features.fit(pd.concat([pd.concat(iv_train), pd.concat(iv_test)]))
    X_train = scaler_features.transform(pd.concat(iv_train))

    regressor = RandomForestRegressor(n_estimators=n_ests, random_state=0)
    regressor.fit(X_train, y_train)
    
    return(regressor, scaler_features)



def predict_with_model(dat, regressor, scaler, iv_test, dv_test, s,  diffs=False):

    test_date_range = dv_test[0].index

    plots = []
    plots_zoom = []
    plots_abs = []
    nums = []

    for i in range(len(facebook_states)):
        X_test = scaler.transform(iv_test[i])
        y_test = dv_test[i]
        y_pred = regressor.predict(X_test)

        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))


        y_pred = pd.DataFrame(np.floor(y_pred))
        y_pred.index = test_date_range
        y_pred.columns = ["Predicted"]

        if(diffs):
            plots.append(hv.Curve(dat[i]["#cases"]).opts(title=facebook_states[i]) * hv.Curve(y_pred).opts(title=facebook_states[i]))
        
        else:
            plots.append(hv.Curve(s[i]["#cases"]).opts(title=facebook_states[i]) * hv.Curve(y_pred).opts(title=facebook_states[i]))

        plots_zoom.append((hv.Curve(s[i]["#cases"].loc[test_date_range]).opts(title=facebook_states[i])* hv.Curve(y_pred).opts(title=facebook_states[i])))


        for j in range(len(y_pred)):
            if(j==0):
                y_pred["Predicted"][j] = y_pred["Predicted"][j] + s[i].loc[test_date_range[0]-pd.Timedelta(days=1),"#cases"]

            y_pred["Predicted"][j] = y_pred["Predicted"][j] + y_pred["Predicted"][j-1]


        y = hv.Div("<div align='top' style = 'margin-left: 50px; font-size: 12px;' >" +
                            "MAE: " + "{:.2f}".format(mae) +
                            ", MSE: " + "{:.2f}".format(mse) +
                            ", RMSE: " + "{:.2f}".format(rmse) +
                            "<div>")

        kennzahl = [mae, mse, rmse]
        plots_abs.append((hv.Curve(s[i]["#cases"]).opts(title=facebook_states[i]) * hv.Curve(y_pred).opts(title=facebook_states[i])))
        nums.append(kennzahl)
          
    return([plots, plots_zoom, plots_abs, nums, regressor, y_pred])