import streamlit_shadcn_ui as ui
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import time
#import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)



### The st.title() function sets the title of the Streamlit application
st.title("Student Dropout Rate In Portugal")


### menu bar

selected = option_menu(
  menu_title = None,
  options = ["Overview","Visualisation","Prediction","Conclusion"],
  icons = ["menu-up", "pie-chart-fill", "graph-up-arrow","recycle"],
  default_index = 0,
  orientation = "horizontal",

)



multiplication = 0
number = st.number_input("Enter a number")
mul1 = multiplication
multiplication = (45 * number)/100
if number >= 0:
    des = f"+{number}% f1 from previous"
else:
    des = f"-{number}% f1 from previous"

difference = (multiplication - mul1)%100

cols = st.columns(3)
with cols[0]:
    ui.metric_card(title="F1-Score", content=f"{multiplication}%", description=des, key="card1")
with cols[1]:
    ui.metric_card(title="Accuracy", content="$45,231.89", description="+20.1% from last month", key="card2")
with cols[2]:
    ui.metric_card(title="Precision", content="$45,231.89", description="+20.1% from last month", key="card3")