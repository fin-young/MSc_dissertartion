#Data Processing
import pandas as pd
import numpy as np
from numpy import hstack
from numpy import array
import datapackage
import requests
from functools import reduce
#ML Packages
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Data APIs
import world_bank_data as wb
import quandl
quandl.ApiConfig.api_key = "bu8h2aStGXk6JxZRbQd9"
# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import plotly.express as px

def Call_IMF(DB,freq,Country_iso2, start, finish, Indicator_code):
    '''
    Returns "Year" & "Values" of series with column names after the IMF Code
    '''
    core = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/'
    key = f'{DB}/{freq}.{Country_iso2}.{Indicator_code}'
    time = f'.?startPeriod={start}&endPeriod={finish}'
    url = f'{core}{key}{time}'

    # Get data from the above URL using the requests package
    data = requests.get(url).json()

    # Load data into a pandas dataframe
    series = pd.DataFrame(data['CompactData']['DataSet']['Series']['Obs'])
    series['@OBS_VALUE'] = series['@OBS_VALUE'].astype(float)
    series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':Indicator_code}, inplace = True)
    return series

def Call_WB(iso3, id,start, finish):
    '''
    Returns "Country"(ISO3), "Year" & "Values" of series with column names after the World Bank Code
    '''
    daterange = f'{start}:{finish}'
    Data = pd.DataFrame(wb.get_series(id, date=daterange, id_or_value='id', simplify_index=True)).reset_index()
    Data = Data[Data['Country']==iso3]
    return Data


def Get_Monthly_GDP(iso3,start, finish):
    '''
    Returns Monthly Call_WB()
    '''
    Data = Call_WB(iso3=iso3, id='NY.GDP.MKTP.CD',start=start, finish=(finish+1))
    Data['Year'] = pd.to_datetime(Data['Year'], exact = False, format='%Y%')
    Data.set_index(['Year'],inplace=True)
    Data.drop(columns='Country', inplace = True)
    Data['GDP % Increase'] = 100*Data['NY.GDP.MKTP.CD'].pct_change()
    Data = Data.resample('MS').ffill() / 12
    Data = Data.reset_index()

    return Data
def Get_Quarterly_GDP(iso3,start, finish):
    '''
    Returns Monthly Call_WB()
    '''
    Data = Call_WB(iso3=iso3, id='NY.GDP.MKTP.CD',start=start, finish=(finish+1))
    Data['Year'] = pd.to_datetime(Data['Year'], exact = False, format='%Y%')
    Data.set_index(['Year'],inplace=True)
    Data.drop(columns='Country', inplace = True)
    Data = Data.resample('QS').ffill() / 4
    Data = Data.reset_index()

    return Data
def Get_Monthly_QEDS(iso3,start, finish):
    '''
    Returns Monthly Call_WB()
    '''
    dfs = [ Call_WB(iso3=iso3, id='DP.DOD.DECN.CR.GG.Z1',start=start, finish=(finish+1)), #Total_Debt
            Call_WB(iso3=iso3, id='DP.DOD.DSTC.CR.GG.Z1',start=start, finish=(finish+1))] #Shrt_Trm_Debt

    Data = reduce(lambda  left,right: pd.merge(left,right,on=['Year', 'Country'],how='outer'), dfs).fillna(np.NaN)
    
    start_date = pd.to_datetime(start, format = '%Y')
    end_date = pd.to_datetime(finish, format = '%Y')

    #Change to quarter start
    Data['Year'] = pd.to_datetime(Data['Year'])
    Data.set_index(['Year'],inplace=True)
    Data = Data.loc[start_date:end_date].copy()
    Data.drop(columns='Country', inplace = True)
    Data = Data.resample('MS').bfill()
    Data = Data.reset_index()

    return Data

def QEDS(iso3,start, finish):
    '''
    Returns quarterly Call_WB()
    '''
    dfs = [ Call_WB(iso3=iso3, id='DP.DOD.DECN.CR.GG.Z1',start=start, finish=(finish+1)), #Total_Debt
            Call_WB(iso3=iso3, id='DP.DOD.DSTC.CR.GG.Z1',start=start, finish=(finish+1))] #Shrt_Trm_Debt

    Data = reduce(lambda  left,right: pd.merge(left,right,on=['Year', 'Country'],how='outer'), dfs).fillna(np.NaN)
    
    start_date = pd.to_datetime(start, format = '%Y')
    end_date = pd.to_datetime(finish, format = '%Y')

    #Change to quarter start
    Data['Year'] = pd.to_datetime(Data['Year'])
    Data.set_index(['Year'],inplace=True)
    Data = Data.loc[start_date:end_date].copy()
    Data['DP.DOD.DECN.CR.GG.Z1'] = Data['DP.DOD.DECN.CR.GG.Z1']*10**6
    Data['DP.DOD.DSTC.CR.GG.Z1'] = Data['DP.DOD.DSTC.CR.GG.Z1']*10**6
    Data.drop(columns='Country', inplace = True)
    return Data


def Call_M_IMF(iso2, start, finish):
    mth_fields = ['RAXGFX_USD','TMG_CIF_USD','TXG_FOB_USD','PCPI_IX',
                '35L___XDC','RAXG_USD','EREER_IX','32____XDC','FILR_PA','FIDR_PA','ENDA_XDC_USD_RATE']
    
    call_string = "+".join(mth_fields)    
    core = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/'
    key = f'IFS/M.{iso2}.{call_string}'
    time = f'.?startPeriod={start}&endPeriod={finish}'
    url = f'{core}{key}{time}'

    # Get data from the above URL using the requests package
    data = requests.get(url).json()
    # Load data into a pandas dataframe
    #series = pd.DataFrame(data['CompactData']['DataSet']['Series']['Obs'])
    appended_data = []
    num_rows = pd.DataFrame(data['CompactData']['DataSet']['Series']).shape[0]
    #print(pd.DataFrame(data['CompactData']['DataSet']['Series']))
    for c in range(num_rows):
        header = data['CompactData']['DataSet']['Series'][c]['@INDICATOR']
        #print(header)
        unit_mult = float(data['CompactData']['DataSet']['Series'][c]['@UNIT_MULT'])
        series = pd.DataFrame(data['CompactData']['DataSet']['Series'][c]['Obs'])
        series['@OBS_VALUE']=series['@OBS_VALUE'].astype(float)
        if unit_mult != 0:
            series['@OBS_VALUE'] = series['@OBS_VALUE']*(10**unit_mult)

        series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':header}, inplace = True)
        series['Year'] = pd.to_datetime(series['Year'], exact = False, format='%Y%')
        appended_data.append(series)

    #Add GDP Data
    WB_countries = wb.get_countries().reset_index()
    iso3 = WB_countries['id'][WB_countries['iso2Code']==iso2].item()
    region = WB_countries['region'][WB_countries['iso2Code']==iso2].item()
    appended_data.append(Get_Monthly_GDP(iso3,start, finish))
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Year'],how='outer'), appended_data).fillna(np.NaN)

    #Add Derived Monthly GDP Date
    

    #Create Derived Fields
    #Real interest rate adjusted to inflation
    if 'FILR_PA' in merged_df.columns and 'PCPI_IX' in merged_df.columns:
        merged_df['Real Interest Rate(%)'] = merged_df['FILR_PA'] - merged_df['PCPI_IX']
    #total reserves (without gold) as % of GDP
    if 'RAXGFX_USD' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns:
        merged_df['Foreign Exchange Reserves(%)'] = 100*(merged_df['RAXGFX_USD']/ (merged_df['NY.GDP.MKTP.CD']))
    #M2 Multiplier Growth
    if '35L___XDC' in merged_df.columns:
        merged_df['M2 Multiplier Growth (%)'] = 100*(merged_df['35L___XDC'].pct_change())
    #'M2/Reserves'
    if '35L___XDC' in merged_df.columns and 'RAXGFX_USD' in merged_df.columns and 'ENDA_XDC_USD_RATE' in merged_df.columns:
        merged_df['M2/Reserves'] = merged_df['35L___XDC']/(merged_df['RAXGFX_USD']*merged_df['ENDA_XDC_USD_RATE'])
    #1 year Deviation of REED
    if 'EREER_IX' in merged_df.columns:
        merged_df['REED 12mth std dev'] = merged_df['EREER_IX'].rolling(12).std()
    #Domestic Credit to GDP
    if '32____XDC' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns and 'ENDA_XDC_USD_RATE' in merged_df.columns:
        merged_df['Domestic Credit to GDP'] = (merged_df['32____XDC']/(merged_df['NY.GDP.MKTP.CD']*merged_df['ENDA_XDC_USD_RATE']))
    #Trade Openness 
    if 'TXG_FOB_USD' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns and 'TMG_CIF_USD' in merged_df.columns:
        merged_df['Trade Openness'] = (merged_df['TXG_FOB_USD']+ merged_df['TMG_CIF_USD'])/(merged_df['NY.GDP.MKTP.CD'])

    #Classify the Data   
    if 'ENDA_XDC_USD_RATE' in merged_df.columns:
        merged_df['% Diff'] = (100*merged_df['ENDA_XDC_USD_RATE'].pct_change(periods = 3)).round(6) 
        #cut_labels = ['normal behaviour', 'Low', 'Medium', 'High']
        cut_labels = [0,1,2,3]
        regions = pd.DataFrame({'region':['East Asia & Pacific', 'Europe & Central Asia','Latin America & Caribbean ', 'South Asia','Sub-Saharan Africa '],
                        'Level_1':[5,10,7.5,5,5],
                        'Level_2':[10,15,12.5,10,10],
                        'Level_3':[15,20,17.5,15,15]
                        })
        regions = regions[regions['region']==region]
        min_move = merged_df['% Diff'].min()
        max_move = merged_df['% Diff'].max()
        if regions['Level_3'][regions['region']==region].item() > max_move:
            max_move = regions['Level_3'][regions['region']==region].item() + 1
        
        cuts = [min_move,
                regions['Level_1'][regions['region']==region].item(),
                regions['Level_2'][regions['region']==region].item(),
                regions['Level_3'][regions['region']==region].item(),
                max_move]
        merged_df['Class'] = pd.cut(merged_df['% Diff'],bins=cuts,labels=cut_labels)
        cond_1 = merged_df['Class'].isna()
        cond_2 = merged_df['% Diff'] <  regions['Level_1'][regions['region']==region].item() 
        merged_df['Class'] = np.where(cond_1 & cond_2,0, merged_df['Class']) # normal behaviour fill

    to_drop = ['RAXGFX_USD','35L___XDC','RAXG_USD','EREER_IX','32____XDC','@OBS_STATUS','% Diff']
    filtered_list = [col for col in to_drop if col  in merged_df.columns.to_list() ]
    merged_df.drop(columns =filtered_list, inplace = True )

    merged_df['days in month']  = pd.to_datetime(merged_df['Year']).dt.to_period('M').dt.days_in_month
    
    merged_df.set_index('Year', inplace = True)
    merged_df = merged_df.resample('D').pad()
    split = ['TMG_CIF_USD','TXG_FOB_USD','M2 Multiplier Growth','GDP % Increase']

    for i in range(len(split)):
        if split[i] in merged_df.columns.to_list():
            merged_df[split[i]] = merged_df[split[i]] / merged_df['days in month']
    merged_df.drop(columns =('days in month'), inplace = True )

    return merged_df


def Call_Q_IMF(iso2,start, finish):
    IFS_fields = ['IAP_BP6_USD','LUR_PT']
    BOP_fields = ['BCAXF_BP6_USD','BFDLXF_BP6_USD']

    IFS_string = "+".join(IFS_fields)
    BOP_string = "+".join(BOP_fields)

    core = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/'
    IFS_key = f'IFS/Q.{iso2}.{IFS_string}'
    BOP_key = f'IFS/Q.{iso2}.{BOP_string}'
    time = f'.?startPeriod={start}&endPeriod={finish}'
    IFS_url = f'{core}{IFS_key}{time}'
    BOP_url = f'{core}{BOP_key}{time}'

    # Get data from the above URL using the requests package
    IFS_data = requests.get(IFS_url).json()
    BOP_data = requests.get(BOP_url).json()
    # Load data into a pandas dataframe
    
    appended_data = []
    
    
    #IFS DATA
    num_rows = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series']).shape[0]
    count_check = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'])
    fields = len(count_check['@INDICATOR'].unique())
    if fields == 1:
        header = (pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'])['@INDICATOR'].unique()).item()
        series = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series']['Obs'])
        unit_mult = float(pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'])['@UNIT_MULT'].unique())
    else:
        for c in range(num_rows):
            header = IFS_data['CompactData']['DataSet']['Series'][c]['@INDICATOR']
            unit_mult = float(IFS_data['CompactData']['DataSet']['Series'][c]['@UNIT_MULT'])
            series = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'][c]['Obs'])
        series['@OBS_VALUE']=series['@OBS_VALUE'].astype(float)
        if unit_mult != 0:
            series['@OBS_VALUE'] = series['@OBS_VALUE']*(10**unit_mult)

        series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':header}, inplace = True)
        series['Year'] = pd.PeriodIndex(series['Year'], freq='Q').to_timestamp()
        appended_data.append(series)
    #BOP DATA
    num_rows = pd.DataFrame(BOP_data['CompactData']['DataSet']['Series']).shape[0]
    #print(pd.DataFrame(BOP_data['CompactData']['DataSet']['Series']))
    for c in range(num_rows):
        header = BOP_data['CompactData']['DataSet']['Series'][c]['@INDICATOR']
        #print(header)
        unit_mult = float(BOP_data['CompactData']['DataSet']['Series'][c]['@UNIT_MULT'])
        series = pd.DataFrame(BOP_data['CompactData']['DataSet']['Series'][c]['Obs'])
        series['@OBS_VALUE']=series['@OBS_VALUE'].astype(float)
        if unit_mult != 0:
            series['@OBS_VALUE'] = series['@OBS_VALUE']*(10**unit_mult)

        series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':header}, inplace = True)
        series['Year'] = pd.PeriodIndex(series['Year'], freq='Q').to_timestamp()
        appended_data.append(series)



    #Add GDP Data
    WB_countries = wb.get_countries().reset_index()
    iso3 = WB_countries['id'][WB_countries['iso2Code']==iso2].item()
    appended_data.append(Get_Quarterly_GDP(iso3,start, finish))
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Year'],how='outer'), appended_data).fillna(np.NaN)

    #Add Derived GDP Date


    #Create Derived Fields
    #net FDI inflows as % of GDP
    if 'BFDLXF_BP6_USD' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns:
        merged_df['FDI as % of GDP(%)'] = 100*(merged_df['BFDLXF_BP6_USD'] / merged_df['NY.GDP.MKTP.CD'])
    #current account balance as % of GDP
    if 'BCAXF_BP6_USD' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns:
        merged_df['Foreign Exchange Reserves(%)'] = 100*(merged_df['BCAXF_BP6_USD']/ merged_df['NY.GDP.MKTP.CD'])


    to_drop = ['BFDLXF_BP6_USD','BCAXF_BP6_USD','NY.GDP.MKTP.CD','EREER_IX','32____XDC','@OBS_STATUS']


    filtered_list = [col for col in to_drop if col  in merged_df.columns.to_list() ]
    filtered_list+=[col for col in merged_df.columns if 'OBS_STATUS' in col]
    merged_df.drop(columns =filtered_list, inplace = True )
    
    #Turn to Monthly & add QEDS Data
    merged_df.set_index('Year', inplace = True)
    merged_df = merged_df.resample('MS').ffill().reset_index()
    if  'IAP_BP6_USD' in merged_df.columns:
        #Linear USD Split to monthly data
        merged_df['IAP_BP6_USD'] = merged_df['IAP_BP6_USD']/3
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Year'],how='outer'), [merged_df,Get_Monthly_QEDS(iso3,start=start, finish=finish)]).fillna(np.NaN)
    
    #Turn Data from monthly to daily
    merged_df['days in month']  = pd.to_datetime(merged_df['Year']).dt.to_period('M').dt.days_in_month
    merged_df.set_index('Year', inplace = True)
    merged_df = merged_df.resample('D').pad()
    if  'IAP_BP6_USD' in merged_df.columns:
        #Linear USD Split to monthly data
        merged_df['IAP_BP6_USD'] = merged_df['IAP_BP6_USD']/ merged_df['days in month']
    merged_df.drop(columns =('days in month'), inplace = True )
    return merged_df

def join_data(iso2,start, finish):
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Year'],how='outer'), [Call_Q_IMF(iso2=iso2,start=start, finish=finish).reset_index(), Call_M_IMF(iso2=iso2,start=start, finish=finish).reset_index()]).fillna(np.NaN)
    merged_df.set_index('Year', inplace = True)
    return merged_df

def Annual_GDP(iso3, start, finish):
    '''
    Returns Annual Call_WB()
    '''
    Data = Call_WB(iso3=iso3, id='NY.GDP.MKTP.CD',start=start, finish=(finish+1))
    Data['Year'] = pd.to_datetime(Data['Year'], exact = False, format='%Y%')
    Data.set_index(['Year'],inplace=True)
    Data.drop(columns='Country', inplace = True)
    Data = Add_datetimes(Data, wave = 'N')
    Data = Data.reset_index()
    return Data
    
def MacEcon_M_TS(iso2, start, finish):
    '''
    in: iso2 = country you want, start = beginning data, finish = final year
    out: DF with Monthly time series of Macro economic features with Year, month num, Day of month) 
    '''
    mth_fields = ['RAXGFX_USD','TMG_CIF_USD','TXG_FOB_USD','PCPI_IX',
                '35L___XDC','RAXG_USD','EREER_IX','32____XDC','FILR_PA','FIDR_PA','ENDA_XDC_USD_RATE']
    
    call_string = "+".join(mth_fields)    
    core = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/'
    key = f'IFS/M.{iso2}.{call_string}'
    time = f'.?startPeriod={start}&endPeriod={finish}'
    url = f'{core}{key}{time}'

    # Get data from the above URL using the requests package
    data = requests.get(url).json()
    # Load data into a pandas dataframe
    #series = pd.DataFrame(data['CompactData']['DataSet']['Series']['Obs'])
    appended_data = []
    num_rows = pd.DataFrame(data['CompactData']['DataSet']['Series']).shape[0]
    #print(pd.DataFrame(data['CompactData']['DataSet']['Series']))
    for c in range(num_rows):
        header = data['CompactData']['DataSet']['Series'][c]['@INDICATOR']
        #print(header)
        unit_mult = float(data['CompactData']['DataSet']['Series'][c]['@UNIT_MULT'])
        series = pd.DataFrame(data['CompactData']['DataSet']['Series'][c]['Obs'])
        series['@OBS_VALUE']=series['@OBS_VALUE'].astype(float)
        if unit_mult != 0:
            series['@OBS_VALUE'] = series['@OBS_VALUE']*(10**unit_mult)

        series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':header}, inplace = True)
        series['Year'] = pd.to_datetime(series['Year'], exact = False, format='%Y%')
        appended_data.append(series)
    monthly_data = reduce(lambda  left,right: pd.merge(left,right,on=['Year'],how='outer'), appended_data).fillna(np.NaN)
    monthly_data.set_index('Year', inplace = True)
    monthly_data = Add_datetimes(monthly_data, wave = 'N')
    monthly_data.drop(columns = ('DoM'), inplace = True)
    
    #Add GDP Data
    WB_countries = wb.get_countries().reset_index()
    iso3 = WB_countries['id'][WB_countries['iso2Code']==iso2].item()
    region = WB_countries['region'][WB_countries['iso2Code']==iso2].item()
    GDP = Annual_GDP(iso3,start, finish)
    GDP.drop(columns = ('MoY'), inplace = True)
    appended_data = [monthly_data, GDP]
    merged_df = reduce(lambda  left,right: pd.merge(left,right,on=['Yr'],how='outer'), appended_data).fillna(np.NaN)
    merged_df.drop(columns = (['DoM']), inplace = True)
    
    # Derived Fields
    #Real interest rate adjusted to inflation
    if 'FILR_PA' in merged_df.columns and 'PCPI_IX' in merged_df.columns:
        merged_df['Real Interest Rate(%)'] = merged_df['FILR_PA'] - merged_df['PCPI_IX']
    #total reserves (without gold) as % of GDP
    if 'RAXGFX_USD' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns:
        merged_df['Foreign Exchange Reserves(%)'] = 100*(merged_df['RAXGFX_USD']/ (merged_df['NY.GDP.MKTP.CD']))
    #M2 Multiplier Growth
    if '35L___XDC' in merged_df.columns:
        merged_df['M2 Multiplier Growth (%)'] = 100*(merged_df['35L___XDC'].pct_change())
    #'M2/Reserves'
    if '35L___XDC' in merged_df.columns and 'RAXGFX_USD' in merged_df.columns and 'ENDA_XDC_USD_RATE' in merged_df.columns:
        merged_df['M2/Reserves'] = merged_df['35L___XDC']/(merged_df['RAXGFX_USD']*merged_df['ENDA_XDC_USD_RATE'])
    #1 year Deviation of REED
    if 'EREER_IX' in merged_df.columns:
        merged_df['REED 12mth std dev'] = merged_df['EREER_IX'].rolling(12).std()
    #Domestic Credit to GDP
    if '32____XDC' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns and 'ENDA_XDC_USD_RATE' in merged_df.columns:
        merged_df['Domestic Credit to GDP'] = (merged_df['32____XDC']/(merged_df['NY.GDP.MKTP.CD']*merged_df['ENDA_XDC_USD_RATE']))
    #Trade Openness 
    if 'TXG_FOB_USD' in merged_df.columns and 'NY.GDP.MKTP.CD' in merged_df.columns and 'TMG_CIF_USD' in merged_df.columns:
        merged_df['Trade Openness'] = (merged_df['TXG_FOB_USD']+ merged_df['TMG_CIF_USD'])/(merged_df['NY.GDP.MKTP.CD'])
    to_drop = ['RAXGFX_USD','35L___XDC','RAXG_USD','EREER_IX','32____XDC','@OBS_STATUS','% Diff']
    filtered_list = [col for col in to_drop if col  in merged_df.columns.to_list() ]
    merged_df.drop(columns =filtered_list, inplace = True )
    
    old_columns = ['TMG_CIF_USD','ENDA_XDC_USD_RATE','PCPI_IX','TXG_FOB_USD','FIDR_PA','FILR_PA']
    new_columns = ['Imports (USD)','Ave Monthly USD FX Rate','Inflation (CPI) %','Exports(USD)','Deposit Interest Rate %','Lending Interest Rate %']
    columns_dict = dict(zip(old_columns, new_columns))
    merged_df.rename(columns = columns_dict, inplace = True)
    
    return merged_df


def MacEcon_Q_TS(iso2,start, finish):
    IFS_fields = ['IAP_BP6_USD','LUR_PT',]# Portfolio Investments - portfolio investment net at current USD, Unemployment Rate
    BOP_fields = ['BCAXF_BP6_USD','BFDLXF_BP6_USD']#Current Account - current account balance as % of GDP, 

    IFS_string = "+".join(IFS_fields)
    BOP_string = "+".join(BOP_fields)

    core = 'http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/'
    IFS_key = f'IFS/Q.{iso2}.{IFS_string}'
    BOP_key = f'IFS/Q.{iso2}.{BOP_string}'
    time = f'.?startPeriod={start}&endPeriod={finish}'
    IFS_url = f'{core}{IFS_key}{time}'
    BOP_url = f'{core}{BOP_key}{time}'

    # Get data from the above URL using the requests package
    IFS_data = requests.get(IFS_url).json()
    BOP_data = requests.get(BOP_url).json()
    # Load data into a pandas dataframe
    
    appended_data = []
    
    
    #IFS DATA
    num_rows = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series']).shape[0]
    count_check = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'])
    fields = len(count_check['@INDICATOR'].unique())
    if fields == 1:
        header = (pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'])['@INDICATOR'].unique()).item()
        series = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series']['Obs'])
        unit_mult = float(pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'])['@UNIT_MULT'].unique())
        series['@OBS_VALUE']=series['@OBS_VALUE'].astype(float)
        if unit_mult != 0:
            series['@OBS_VALUE'] = series['@OBS_VALUE']*(10**unit_mult)
        series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':header}, inplace = True)
        series['Year'] = pd.PeriodIndex(series['Year'], freq='Q').to_timestamp()
        appended_data.append(series)
    else:
        for c in range(num_rows):
            header = IFS_data['CompactData']['DataSet']['Series'][c]['@INDICATOR']
            unit_mult = float(IFS_data['CompactData']['DataSet']['Series'][c]['@UNIT_MULT'])
            series = pd.DataFrame(IFS_data['CompactData']['DataSet']['Series'][c]['Obs'])
            series['@OBS_VALUE']=series['@OBS_VALUE'].astype(float)
            if unit_mult != 0:
                series['@OBS_VALUE'] = series['@OBS_VALUE']*(10**unit_mult)

            series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':header}, inplace = True)
            series['Year'] = pd.PeriodIndex(series['Year'], freq='Q').to_timestamp()
            appended_data.append(series)
    #BOP DATA
    num_rows = pd.DataFrame(BOP_data['CompactData']['DataSet']['Series']).shape[0]
    #print(pd.DataFrame(BOP_data['CompactData']['DataSet']['Series']))
    for c in range(num_rows):
        header = BOP_data['CompactData']['DataSet']['Series'][c]['@INDICATOR']
        unit_mult = float(BOP_data['CompactData']['DataSet']['Series'][c]['@UNIT_MULT'])
        series = pd.DataFrame(BOP_data['CompactData']['DataSet']['Series'][c]['Obs'])
        series['@OBS_VALUE']=series['@OBS_VALUE'].astype(float)
        if unit_mult != 0:
            series['@OBS_VALUE'] = series['@OBS_VALUE']*(10**unit_mult)

        series.rename(columns = {'@TIME_PERIOD':'Year', '@OBS_VALUE':header}, inplace = True)
        series['Year'] = pd.PeriodIndex(series['Year'], freq='Q').to_timestamp()
        appended_data.append(series)

    quarterly_data = reduce(lambda  left,right: pd.merge(left,right,on=['Year'],how='outer'), appended_data).fillna(np.NaN)

    #QEDS Data
    WB_countries = wb.get_countries().reset_index()
    iso3 = WB_countries['id'][WB_countries['iso2Code']==iso2].item()
    QEDS_data = QEDS(iso3,start=start, finish=finish)
    appended_data = [quarterly_data, QEDS_data]
    quarterly_data = reduce(lambda  left,right: pd.merge(left,right,on=['Year'],how='outer'), appended_data).fillna(np.NaN)
    quarterly_data.set_index('Year', inplace = True)
    quarterly_data = Add_datetimes(quarterly_data, wave = 'N')
    GDP = Annual_GDP(iso3,start=start, finish=finish)
    GDP.drop(columns = (['MoY','DoM']), inplace = True)
    quarterly_data = pd.merge(quarterly_data,GDP,on=['Yr'],how='outer').fillna(np.NaN)
    
    # Reform column names
    quarterly_data['BCAXF_BP6_USD'] = 100*(quarterly_data['BCAXF_BP6_USD']/quarterly_data['NY.GDP.MKTP.CD'])
    quarterly_data['BFDLXF_BP6_USD'] = 100*(quarterly_data['BFDLXF_BP6_USD']/quarterly_data['NY.GDP.MKTP.CD'])
    old_columns = ['IAP_BP6_USD','LUR_PT','BCAXF_BP6_USD','BFDLXF_BP6_USD','DP.DOD.DECN.CR.GG.Z1','DP.DOD.DSTC.CR.GG.Z1']
    new_columns = ['Portfolio Investments(USD)','Umemployment rate(%)','Current Account(%GDP)','FDI(%GDP)','Total Debt(USD)','Short Term Debt(USD)']
    columns_dict = dict(zip(old_columns, new_columns))
    quarterly_data.rename(columns = columns_dict, inplace = True)

    # remove status columns 
    filtered_list=[col for col in quarterly_data.columns if 'OBS_STATUS' in col]
    filtered_list.append('NY.GDP.MKTP.CD')
    quarterly_data.drop(columns =filtered_list, inplace = True )    
    return quarterly_data

def MacEcon_A_TS(iso2,start, finish):
    '''
    Returns annual Call_WB()
    '''
    WB_countries = wb.get_countries().reset_index()
    iso3 = WB_countries['id'][WB_countries['iso2Code']==iso2].item()
    
    dfs = [ Call_WB(iso3=iso3, id='NE.GDI.FTOT.CD',start=start, finish=(finish+1)), #Gross fixed capital formation (current US$)
            Call_WB(iso3=iso3, id='NE.CON.GOVT.ZS',start=start, finish=(finish+1)), #General government final consumption expenditure as % of GDP
            Call_WB(iso3=iso3, id='NY.GDP.MKTP.CD',start=start, finish=(finish+1))]
    
    Data = reduce(lambda  left,right: pd.merge(left,right,on=['Year', 'Country'],how='outer'), dfs).fillna(np.NaN)
    
    start_date = pd.to_datetime(start, format = '%Y')
    end_date = pd.to_datetime(finish, format = '%Y')

    #Change to quarter start
    Data['Year'] = pd.to_datetime(Data['Year'])
    Data.set_index(['Year'],inplace=True)
    Data = Data.loc[start_date:end_date].copy()
    Data.drop(columns='Country', inplace = True)
    Data = Add_datetimes(Data, wave = 'N')    
    
    # Reform column names
    Data['NY.GDP.MKTP.CD'] = 100*(Data['NY.GDP.MKTP.CD'].pct_change())
    old_columns = ['NE.GDI.FTOT.CD','NE.CON.GOVT.ZS','NY.GDP.MKTP.CD']
    new_columns = ['Capital Formations(USD)','Gov Consumption Expendature (%GDP)','GDP % Growth']
    columns_dict = dict(zip(old_columns, new_columns))
    Data.rename(columns = columns_dict, inplace = True)
    
    return Data
    
def MacEcon_TS(iso2,start, finish):  
    '''
    Joing monthly, Quarterly, anuually available data with one another. 
    '''
    Mth=MacEcon_M_TS(iso2=iso2,start=start, finish=finish)
    Qrt=MacEcon_Q_TS(iso2=iso2,start=start, finish=finish)
    Ann=MacEcon_A_TS(iso2=iso2,start=start, finish=finish)
    fills = Qrt.columns.to_list() + Ann.columns.to_list()
    exc = ['DoM','Yr','Year','MoY']
    fill_list = [col for col in fills if col  not in exc ]
    #Ann.drop(columns=['MoY'], inplace = True)
    
    Qrt_Mth =pd.merge(Mth,Qrt,on=['Yr', 'MoY'],how='outer').fillna(np.NaN)
    Full = pd.merge(Qrt_Mth,Ann,on=['Yr', 'MoY'],how='left').fillna(np.NaN)
    filtered_list=[col for col in Full.columns if 'DoM' in col]
    filtered_list+=[col for col in Full.columns if 'Year' in col]
    Full.drop(columns =filtered_list, inplace = True )
    Full.sort_values(by=['Yr', 'MoY'], inplace = True)
    Full = Full.iloc[:-1 , :]
    Full['MoY']=Full['MoY'].astype(int)
    Full['Date']=pd.to_datetime([f'{y}-{m}-01' for y, m in zip(Full.Yr, Full.MoY)])
    
    preceed = ['Date','Yr','MoY']
    the_rest = [col for col in Full.columns if col  not in preceed ]
    new_order = preceed + the_rest
    Full = Full.reindex(columns=new_order)
    for col in fill_list:
        Full[col] = Full[col].ffill()
    Full.set_index(['Date'], inplace = True)
    
    return Full

def generate_cyclical_features(df, col_name, period, start_num=0, drop = 'Y'):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/df[period]),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/df[period])    
             }
    if drop == 'Y':
        df = df.assign(**kwargs).drop(columns=[col_name, period])
    elif drop == 'N':
        df = df.assign(**kwargs)
    return df

def cyclical_data(dataset):
    A = dataset.reset_index().copy()
    idx = A.columns[0]
    period = pd.to_datetime(A[idx]).dt.to_period('D')
    A['Yr'] = period.dt.year
    A['leap'] = period.dt.is_leap_year
    A['day/yr'] = np.where(A['leap']==True, 366,365)
    A['day/M'] = period.dt.daysinmonth
    A['M/yr'] = period.dt.daysinmonth
    A['DoM'] = period.dt.day
    A['MoY'] = period.dt.month
    A['DoY'] = period.dt.dayofyear

    A = generate_cyclical_features(A,'DoY','day/yr',1,drop = 'Y')
    A = generate_cyclical_features(A, 'MoY','M/yr',1,drop = 'Y')
    A = generate_cyclical_features(A, 'DoM','day/M',1,drop = 'Y')
    A.drop(columns=['Yr', 'leap'], inplace= True)
    dataset = A.copy()
    dataset.set_index(idx,inplace = True)
    return dataset

def Add_datetimes(dataset, wave = 'N'):
    A = dataset.reset_index().copy()
    idx = A.columns[0]
    period = pd.to_datetime(A[idx]).dt.to_period('D')
    A['Yr'] = period.dt.year
    
    A['DoM'] = period.dt.day
    A['MoY'] = period.dt.month
    
    if wave == 'Y':
        A['leap'] = period.dt.is_leap_year
        A['DoY'] = period.dt.dayofyear
        A['day/yr'] = np.where(A['leap']==True, 366,365)
        A = generate_cyclical_features(A,'DoY','day/yr',1,drop = 'Y')
        A.drop(columns=['leap'], inplace= True)
    A.set_index(idx,inplace = True)

    return A

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()
    

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def Get_curr_code(iso2):
    data_url = 'https://datahub.io/core/country-codes/datapackage.json'
    # to load Data Package into storage
    package = datapackage.Package(data_url)
    # to load only tabular data
    resources = package.resources
    for resource in resources:
        if resource.tabular:
            comp_countries = pd.read_csv(resource.descriptor['path'])
    X = comp_countries[['ISO3166-1-Alpha-2','ISO4217-currency_alphabetic_code']]
    curr_code = X['ISO4217-currency_alphabetic_code'][X['ISO3166-1-Alpha-2']==iso2].item()
    return curr_code

def Get_curr_data(iso2):
    curr_code = Get_curr_code(iso2)
    FRED = ['BRL','HKD','INR','KRW','MYR','MXN','ZAR','SGD','LKR','THB','GBP','VEF']
    BOE = ['CZK','HUF','NIS','RUB','TRY']

    if curr_code in FRED: 
        fred_rates = pd.DataFrame({'Currency': {'DEXBZUS': 'Brazilian Real (BRL)',
                                                'DEXHKUS': 'Hong Kong Dollar (HKD)',
                                                'DEXINUS': 'Indian Rupee (INR)',
                                                'DEXKOUS': 'South Korean Won (KRW)',
                                                'DEXMAUS': 'Malaysian Ringgit (MYR)',
                                                'DEXMXUS': 'Mexican Peso (MXN)',
                                                'DEXSFUS': 'South African Rand(ZAR)',
                                                'DEXSIUS': 'Singapore Dollar (SGD)',
                                                'DEXSLUS': 'Sri Lankan Rupee(LKR)',
                                                'DEXTHUS': 'Thai Baht (THB)',
                                                'DEXUSUK': 'British Pound (GBP)',
                                                'DEXVZUS': 'Venezuelan Bolivar (VEF)'}})
        fred_rates['symbol'] = fred_rates.Currency.map(lambda x: x[-4:-1])
        fred_rates = fred_rates[fred_rates['symbol']==curr_code]
        rates1 = [quandl.get("FRED/{0}".format(fx)) for fx in fred_rates.index]
        fx_rates = pd.concat(rates1, axis=1)
        fx_rates.columns = [fx for fx in fred_rates.symbol]



    elif curr_code in BOE:
        BOE_rates = pd.DataFrame({'Currency': {'XUDLBK27': 'Czech Koruna (CZK)',
                                               'XUDLBK35': 'Hungarian Forint (HUF)',
                                               'XUDLBK65': 'Israeli Shekel (NIS)',
                                               'XUDLBK69': 'Russian Ruble (RUB)',
                                               'XUDLBK75': 'Turkish Lira (TRY)',
                                                }})
        BOE_rates['symbol'] = BOE_rates.Currency.map(lambda x: x[-4:-1])
        BOE_rates = BOE_rates[BOE_rates['symbol']==curr_code]
        rates2 = [quandl.get("BOE/{0}".format(fx)) for fx in BOE_rates.index]
        fx_rates = pd.concat(rates2, axis=1)
        fx_rates.columns = [fx for fx in BOE_rates.symbol]
    #fx_rates[curr_code] = 1/fx_rates[curr_code]
    return fx_rates

def generate_time_lags(df, n_lags):
    df_n = df.copy()
    col = df.columns.item()
    for n in range(1, n_lags + 1):
        df_n[f"{col}:T-{n}"] = df_n[col].shift(n)
    df_n = df_n.iloc[n_lags:]
    return df_n
def Get_FX_MacEcon_Data(iso2):
    FX_Rates = Get_curr_data(iso2)
    FX_Rates = Add_datetimes(FX_Rates, wave = 'Y')
    FX_Rates = FX_Rates.reset_index()
    max_yr = FX_Rates['Date'].max().year
    min_yr = FX_Rates['Date'].min().year
    strt = 2020 if max_yr > 2020 else max_yr
    fin = 1990 if min_yr < 1990 else min_yr
    #needs to be change to dynamic dates
    Mac_Econ_data = MacEcon_TS(iso2=iso2, start=1980, finish=2020)
    dataset = pd.merge(FX_Rates, Mac_Econ_data, on=['Yr','MoY'], how = 'inner')
    dataset.set_index(['Date'], inplace = True)
    #Change FX rate to 90 Rolling average
    fx_field = dataset.iloc[:,0].name
    roll30 = fx_field+'_30_ave'
    roll60 = fx_field+'_60_ave'
    roll90 = fx_field+'_90_ave'
    dataset[roll30] = dataset[fx_field].rolling(30).mean()
    dataset[roll60] = dataset[fx_field].rolling(60).mean()
    dataset[roll90] = dataset[fx_field].rolling(90).mean()
    dataset.dropna(axis=0, subset=[roll90], inplace = True)
    return dataset

def describe_MacEcon_TS(df):

    columns = df.columns
    firsts = []
    lasts = []
    nans = []
    for i in range(len(columns)):
        col = columns[i]
        firsts.append(df[[col]].first_valid_index())
        lasts.append(df[[col]].last_valid_index())
        nans.append(df[[col]].isna().sum().item())
    DF = pd.DataFrame({'Features':columns, 'First Valid':firsts, 'Last Valid':lasts, 'Null Values':nans})
    return DF

def Add_dummy_MoY(df):
    dummy = pd.get_dummies(df['MoY'])
    #cols = dummy.columns.to_numpy()
    #cols = np.array_str(cols)
    cols = []
    for i in range(len(dummy.columns)):
        cols.append(f"MoY_is_{i+1}")
    
    columns_dict = dict(zip(dummy.columns, cols))
    dummy.rename(columns = columns_dict, inplace = True)
    df.drop(columns=['MoY'],inplace=True)
    df = df.merge(dummy, left_index=True, right_index=True)
    return df

def shift_FX(dataframe):
    df = dataframe.copy()
    Y_name = df.iloc[:,0].name
    X_data = df[Y_name].shift(1)
    X_name = Y_name+"_X"
    new_name = Y_name+"_Y"
    df[new_name] = df[Y_name].copy()    
    df.rename(columns= {Y_name: X_name}, inplace = True)
    df[X_name] = X_data
    df = df.iloc[1: , :]   
    return df
def stack_sequences(df):
    n_cols = df.shape[1]
    n_rows = df.shape[0]
    appended_series = []
    for col in range(n_cols):
        series = df.iloc[:,col].to_numpy().reshape(n_rows, 1)
        appended_series.append(series)
    stacked_array = hstack((appended_series))
    return stacked_array

def shift_and_stack(df):
    dataframe = shift_FX(df)
    array = stack_sequences(dataframe)
    return array