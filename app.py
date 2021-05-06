import streamlit as st
import math
import numpy as np
import streamlit.components.v1 as components
from flask import Flask, request, jsonify, render_template
from model import Model



#def home():
    #return render_template('login.html')

def app_run():
    """
    Returns
    -------
    Run App
    
    """
    components.html("<head>
  <meta charset="UTF-8">
  <title>Login to RealEstate</title>

	<script src="https://www.gstatic.com/firebasejs/5.0.4/firebase.js"></script>
	<!--
	  var config = {
	    apiKey: "AIzaSyA1rRG77hpoYS3i6moo900KD8GR1wQ5Kh8",
	    authDomain: "okcredit-7535e.firebaseapp.com",
	    databaseURL: "https://okcredit-7535e.firebaseio.com",
	    projectId: "okcredit-7535e",
	    storageBucket: "okcredit-7535e.appspot.com",
	    messagingSenderId: "870984863712"
	  };
	  firebase.initializeApp(config);
	--> <script> var config = {
apiKey: "AIzaSyCleAdT5AyqQR8LDibRw6874LxogcpTUks",
  authDomain: "realestate-34d31.firebaseapp.com",
  projectId: "realestate-34d31",
  storageBucket: "realestate-34d31.appspot.com",
  messagingSenderId: "552651165843",
  appId: "1:552651165843:web:86367251ac1547fdd77d66",
  measurementId: "G-98EG4HWFGX"
};
 firebase.initializeApp(config);
</script>

  <script src="https://cdn.firebase.com/libs/firebaseui/2.3.0/firebaseui.js"></script>
  <link type="text/css" rel="stylesheet" href="https://cdn.firebase.com/libs/firebaseui/2.3.0/firebaseui.css" />
  <!--<link href="style.css" rel="stylesheet" type="text/css" media="screen" />-->
<link href="/static/style.css" rel="stylesheet" type="text/css" media="screen" />

</head>
<body style = "background-color:#6f70a6">
    <div id="container">
	    <h1><u>Login To RealEstate</u></h1>
      <div id="loading">Loading...</div>
      <div id="loaded" class="hidden">
        <div id="main">
          <div id="user-signed-in" class="hidden">
            <div id="user-info">
              <div id="phone">

</div>
              <div class="clearfix"></div>
            </div>
            <p>
              <button id="sign-out">Sign Out</button>

   </a>         </p>
	      <form action="/realapp" class="hidden">
 <button type="submit" class="btn-large waves-effect waves-light orange">Proceed to Webapp</button>
</form>
          </div>
          <div id="user-signed-out" class="hidden">
            <div id="firebaseui-spa">
              <h3>App:</h3>
              <div id="firebaseui-container"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <script src="/static/app.js"></script>
</body>")
    #load Module class
    # Create a text element and let the reader know the data is loading.
                    
def realapp():                    
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
    st.title('Evaluate your home in Telangana')
    
    st.subheader("Discover the market value of your home convenient and easy with one click")
    
    st.image('./data/Header Varese.jpg',use_column_width=True)
    
    st.subheader ("We use a Machine Learning algorithm on %s properties"
         % len(df))
    
    #Parameters
    st.subheader("Set the property parameters")
    
    #City
    city = st.selectbox('City',['Warangal','Suryapet'],index = 0)
    
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
        st.write("₹{:,.0f}".format(value))
        
        if parking_space > 0:
            
            st.write("Average price  %s - %s " % (city,district))
            st.write("₹{:,.0f}".format(parking_space))
        
        
        
    
if __name__ == "__main__":
    #app.run()
    app_run()
    
    
