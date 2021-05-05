import streamlit as st
import math
from model import Model


def app_run():
    """
    Returns
    -------
    Run App
    
    """
    #load Module class
    # Create a text element and let the reader know the data is loading.
    
    with st.spinner(text='In progress'):
    #data_load_state = st.text('Loading data & model...')
    # Notify the reader that the data was successfully loaded.
        __model = Model()
        st.success('Model Ready')
    
    #Dataset & zones
    df = __model.dataset_preprocessed
    
    #Parking Space Price
    parking_price_df =(
        df.loc[df['parkingSpacePrice']>1000].
            groupby('district').mean()['parkingSpacePrice'])
    
    #Title & Header Image
    st.title('Evaluate your home in Hyderabad')
    
    st.subheader("Discover the market value of your home convenient and easy with one click")
    
    st.image('./data/Header Varese.jpg',use_column_width=True)
    
    st.subheader ("We use a Machine Learning algorithm on %s properties"
         % len(df))
    
    #Parameters
    st.subheader("Set the property parameters")
    
    #City
    city = st.selectbox('City',['Warangal','SURYAPET'],index = 0)
    
    selector_city = 'wG - '
    
    if city == 'warangal':
        selector_city = 'WG - '
        
    else:
        selector_city = 'SU - '
    
    #Zone
    zonesList = (
        df.loc[df['district'].str.startswith (selector_city)]['district'].unique().tolist()
        )
    
    #Replace prefix
    zonesList = [i.replace(selector_city,'') for i in zonesList]
    
    district = st.selectbox('Zone', zonesList, index = 0)
    
    #Property Type List
    propertyTypelist = __model.propertyTypeList

    propertyType = st.selectbox('Kind', propertyTypelist, index = 0)
    
    #Conversiont to model variables
    propertyType = __model.propertyTypeConverter(propertyType)
    
    #Rest of parameters  
    size = st.number_input('Square meters',
              min_value=10, 
              max_value=5000,
              value = 100
              )
    
    rooms = st.slider('Locals',
            min_value = 1,
            max_value =  10,
            value = 3)
        
    #Conversiont to model variables
    #roomsCat = __model.roomsCategory(rooms)
    if rooms >= 4:
        roomsCat = 4
    else:
        roomsCat = rooms

    #Bathrooms
    bathrooms = st.slider('Bathrooms',
            min_value = 1,
            max_value = 10,
            value = 2
                          )
    #Conversiont to model variables
    if bathrooms >= 2:
            bathroomsCat = 2
    else:
        bathroomsCat = bathrooms

    #Status italiano
    status_it = __model.statusList
    
    status = st.radio('State',status_it, index = 1)
    
    #Conversiont to model variables
    statusOutput = 'good'

    if status == "To be restructured":
        
        statusOutput = 'renew'
        
    elif status == "Good":
        
        statusOutput = 'good'
        
    elif status == "New Construction ":
    
        statusOutput = 'newdevelopment'
    #Extra Feautures
    #parkingBox = st.checkbox('Posto Auto - Box', value = 0)
        
    #garden = st.checkbox('Giardino- Terrazzo', value = 0)
    
    #swimming_pool = st.checkbox('Piscina', value = 0)
    
    #Parking Space District Selected
    
    try:
    
        parking_space = int(
            parking_price_df.loc[parking_price_df.index==
                                 
                                 (selector_city+district)].values[0])
        
    except:
        
        parking_space = 0
            
    #Button to value    
    button = st.button('Predit Cost')
        
    if button:
        
        value_model = __model.predict( 
            size,
            propertyType,
            (selector_city+district),
            statusOutput,
            roomsCat,
            bathroomsCat,
            ) 
    
        value = int(math.ceil((value_model ) / 5000.0) * 5000.0)
        
        st.write("Market value")
        st.write("{:,.0f}₹".format(value))
        
        if parking_space > 0:
            
            st.write("Average price  %s - %s " % (city,district))
            st.write("{:,.0f}₹".format(parking_space))
        
        
        
    
if __name__ == "__main__":
    
    app_run()
    
