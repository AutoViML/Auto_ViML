######### NEW And FAST WAY to ADD Feature Engg COLUMNS IN A DATA SET #######
###   Modify Dataframe by adding Features using Feature Tools ####
def add_features( df, add_types=[], idcolumn=''):
    import featuretools as ft
    df = copy.deepcopy(df)
    projectid = 'project_prediction'
    dataid = 'project_data'
    if idcolumn == '':
        indexid = 'index'
        make_index = True
    else:
        indexid = idcolumn
        make_index=False
    # Make an entityset and add the entity
    es = ft.EntitySet(id = projectid)
    es.entity_from_dataframe(entity_id = dataid, dataframe = df, 
                             make_index = make_index, index = indexid)

    # Run deep feature synthesis with given input primitives or automatically deep 2
    if len(add_types) > 0:
        ### Build Features based on given primitive types, add_types which is a list
        df_mod, feature_defs = ft.dfs(entityset = es, target_entity = dataid,
                             trans_primitives = add_types)
    else:
        ### Perform Deep Feature Synthesis Automatically for Depth 2
        df_mod, feature_defs = ft.dfs(entityset=es, target_entity = dataid, 
                              max_depth = 3, 
                              verbose = 0)
    if make_index:
        df_mod = df_mod.reset_index(drop=True)
    return df_mod

def feature_engineering(df, ft_requests, idcol):
    """
    The Feature Engineering module needs FeatureTools installed to work. 
    It takes a given data set, df and adds features based on the requet types in
    ft_requests which can be 'add','subtract','multiply','divide'. If you have
    an id_column in the data set, you can provide it as idcol (a string variable).
    It will return your modified dataset with 'idcol' as the index. Make sure
    you reset the index if you want to return it to its former state.
    """
    df = copy.deepcopy(df)
    ft_dict = dict(zip(['add','multiply','subtract','divide'],
                  ['add_numeric', 'multiply_numeric',
            'subtract_numeric', 'divide_numeric']))
    if len(ft_requests) > 0:
        ft_list = []
        for ft_one in ft_requests:
            if ft_one in ft_dict.keys():
                ft_list.append(ft_dict[ft_one])
            else:
                print('    Cannot perform %s-type feature engineering...' %ft_one)    
        cols = [x for x in df.columns.tolist() if x not in [idcol]]
        for each_ft, count in zip(ft_list, range(len(ft_list))):
            if count == 0:
                df_mod = add_features(df,[each_ft], idcol)
                print(df_mod.shape)
            else:
                df_temp = add_features(df,[each_ft], idcol)
                df_temp.drop(cols,axis=1,inplace=True)
                df_mod = pd.concat([df_mod,df_temp],axis=1,ignore_index=False)
                print(df_mod.shape)
    else:
        df_mod = add_features(df,[], idcol)
    return df_mod


def start_end_date_time_processing(smalldf, startTime, endTime, splitter_date_string="/",splitter_hour_string=":"):
    """
    If you have start date time stamp and end date time stamp, this module will create additional features for such fields.
    You must provide a start date time stamp field and if you have an end date time stamp field, you must use it. 
    Otherwise, you are better off using the create_date_time_features module which is also in this library.
    You must provide the following:
    smalldf: Dataframe containing your date time fields
    startTime: this is hopefully a string field which converts to a date time stamp easily. Make sure it is a string.
    endTime: this also must be a string field which converts to a date time stamp easily. Make sure it is a string.
    splitter_date_string: usually there is a string such as '/' or '.' between day/month/year etc. Default is assumed / here.
    splitter_hour_string: usually there is a string such as ':' or '.' between hour:min:sec etc. Default is assumed : here.
    """
    smalldf = smalldf.copy()
    add_cols = []
    start_date = 'processing'+startTime+'_start_date'
    smalldf[start_date] = smalldf[startTime].map(lambda x: x.split(" ")[0])
    add_cols.append(start_date)
    try:
        start_time = 'processing'+startTime+'_start_time'
        smalldf[start_time] = smalldf[startTime].map(lambda x: x.split(" ")[1])
        add_cols.append(start_time)
    except:
        ### there is no hour-minutes part of this date time stamp field. You can just skip it if it is not there
        pass
    end_date = 'processing'+endTime+'_end_date'
    smalldf[end_date] = smalldf[endTime].map(lambda x: x.split(" ")[0])
    add_cols.append(end_date)
    try:
        end_time = 'processing'+endTime+'_end_time'
        smalldf[end_time] = smalldf[endTime].map(lambda x: x.split(" ")[1])
        add_cols.append(end_time)
    except:
        ### there is no hour-minutes part of this date time stamp field. You can just skip it if it is not there
        pass
    view_days = 'processing'+startTime+'_elapsed_days'
    smalldf[view_days] = (pd.to_datetime(smalldf[end_date]) - pd.to_datetime(smalldf[start_date])).values.astype(int)
    add_cols.append(view_days)
    try:
        view_time = 'processing'+startTime+'_elapsed_time'
        smalldf[view_time] = (pd.to_datetime(smalldf[end_time]) - pd.to_datetime(smalldf[start_time])).astype('timedelta64[s]').values
        add_cols.append(view_time)
    except:
        ### In some date time fields this gives an error so skip it in that case
        pass
    #### The reason we chose endTime here is that startTime is usually taken care of by another library. So better to do this alone.
    year = 'processing'+endTime+'_end_year'
    smalldf[year] = smalldf[end_date].map(lambda x: str(x).split(splitter_date_string)[0]).values
    add_cols.append(year)
    #### The reason we chose endTime here is that startTime is usually taken care of by another library. So better to do this alone.
    month = 'processing'+endTime+'_end_month'
    smalldf[month] = smalldf[end_date].map(lambda x: str(x).split(splitter_date_string)[1]).values
    add_cols.append(month)
    try:
        #### The reason we chose endTime here is that startTime is usually taken care of by another library. So better to do this alone.
        daynum = 'processing'+endTime+'_end_day_number'
        smalldf[daynum] = smalldf[end_date].map(lambda x: str(x).split(splitter_date_string)[2]).values
        add_cols.append(daynum)
    except:
        ### In some date time fields the day number is not there. If not, just skip it ####
        pass
    #### In some date time fields, the hour and minute is not there, so skip it in that case if it errors!
    try:
        start_hour = 'processing'+startTime+'_start_hour'
        smalldf[start_hour] = smalldf[start_time].map(lambda x: str(x).split(splitter_hour_string)[0]).values
        add_cols.append(start_hour)
        start_min = 'processing'+startTime+'_start_hour'
        smalldf[start_min] = smalldf[start_time].map(lambda x: str(x).split(splitter_hour_string)[1]).values
        add_cols.append(start_min)
    except:
        ### If it errors, skip it
        pass
    #### Check if there is a weekday and weekends in date time columns using endTime only
    weekday_num = 'processing'+endTime+'_end_weekday_number'
    smalldf[weekday_num] = pd.to_datetime(smalldf[end_date]).dt.weekday.values
    add_cols.append(weekday_num)
    weekend = 'processing'+endTime+'_end_weekend_flag'
    smalldf[weekend] = smalldf[weekday_num].map(lambda x: 1 if x in[5,6] else 0)
    add_cols.append(weekend)
    #### If everything works well, there should be 13 new columns added by module. All the best!
    print('%d columns added using start date=%s and end date=%s processing...' %(len(add_cols),startTime,endTime))
    return smalldf
#
#################################################################################
def split_one_field_into_many(df, field, splitter, filler, new_names_list):
    """
    This little function takes any data frame field (string variables only) and splits
    it into as many fields as you want in the new_names_list.
    You can also specify what string to split on using the splitter argument.
    You can also fill Null values that occur due to your splitting by specifying a filler.
    if no new_names_list is given, then we use the name of the field itself to split.
    """
    import warnings
    warnings.filterwarnings("ignore")
    df = df.copy()
    ### First print the maximum number of things in that field
    max_things = df[field].map(lambda x: len(x.split(splitter))).max()
    print('    Maximum number of columns created using variable %s = %d' %(field,max_things))
    ### This creates a new field that counts the number of things that are in that field. 
    num_products_viewed = 'count_things_in_'+field
    df[num_products_viewed] = df[field].map(lambda x: len(x.split(";"))).values
    ### Clean up the field such that it has the right number of split chars otherwise add to it
    df[field] = df[field].map(lambda x: x+split_char*(max_things-len(x.split(";"))) if len(x.split(";")) < max_things else x)
    ###### Now you create new fields by split the one large field ########
    if new_names_list == '':
        new_names_list = [field+'_'+str(i) for i in range(1,max_things+1)]
    try:
        for i in range(len(new_names_list)):
            df[field].fillna(filler, inplace=True)
            df.loc[df[field] == splitter, field] = filler
            df[new_names_list[i]] = df[field].map(lambda x: x.split(splitter)[i]
                                          if splitter in x else x)
    except:
        ### Check if the column is a string column. If not, give an error message.
        print('Cannot split the column. Getting an error. Check the column again')
        return df
    return df, new_names_list
#################################################################################
