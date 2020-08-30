#this is a python script to import as extension in ipynb 
#the functions perform transformation on the data used for timeseries analysing 


import pandas as pd
from datetime import datetime, timedelta



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


def shift_days(df, days):
    fin=df       
    for i in range(1,days+1):
        dfy = df.drop("AnzahlFall", axis=1).shift(i)
        dfy.columns = [ s + " t-" + str(i) for s in dfy.columns]
        fin = pd.concat([fin, dfy], axis=1)
    return(fin)


def join_dat(tiles, apple, google, rki):
    df = tiles.join(apple).join(google).join(rki)
    df["AnzahlFall"] = df["AnzahlFall"].fillna(0)
    return df
    