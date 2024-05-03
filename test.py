import streamlit_shadcn_ui as ui
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import graphviz
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import f1_score, r2_score,accuracy_score, precision_score,recall_score
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier,export_graphviz # Import Decision Tree Classifier
import mlflow
from sklearn import metrics
from codecarbon import EmissionsTracker
#%matplotlib inline

# Initialize the emissions tracker
tracker = EmissionsTracker()
tracker.start()
#st.set_page_config(layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)
#####################################################################
# Load and cleaning the dataset
df = pd.read_csv("Students.csv")
df_VIZ= pd.read_csv("Student_modified.csv")

img_importance = Image.open('feature_importance.png')
img_importance_subset = Image.open('feature_subset.png')
img_contribution_subset = Image.open('contribution subset.png')
#Renaming the column 'Nacionality' to 'Nationality' and 'Output' to 'Student Status'

df.rename(columns = {'Nacionality':'Nationality', 'Output': 'Student Status'}, inplace = True)


# function that takes categorical values and turns them into corresponding strings.
def cat_to_string(df, column_name, mapping_dict):
    df_string = df.copy()
    # Replace the numbers in the specified column with strings using the map function
    df_string[column_name] = df_string[column_name].map(lambda x: mapping_dict[x] if x in mapping_dict else x)
    return df_string

# Dictionary to map the values in the 'school' column to strings
marital_status_mapping = {
    1: "single",
    2: "married",
    3: "widower",
    4: "divorced",
    5: "facto union",
    6: "legally separated"
}

# application dic
application_mode_mapping = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}

# application order '

application_order_mapping = {
    0: "first choice",
    1: "second choice",
    2: "third choice",
    3: "fourth choice",
    4: "fifth choice",
    5: "sixth choice",
    6: "seventh choice",
    7: "eighth choice",
    8: "ninth choice",
    9: "last choice"
}


# course mapping

course_mapping = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}


# previous qualifications

previous_qualification_mapping = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year) or equivalent",
    38: "Basic education 2nd cycle (6th/7th/8th year) or equivalent",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)"
}

nationality_mapping = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

mothers_qualification_mapping = {
    1: "Secondary Education - 12th Year of Schooling or Equivalent",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent",
    22: "Technical-professional course",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without having a 4th year of schooling",
    37: "Basic education 1st cycle (4th/5th year) or equivalent",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equivalent",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)"
}

fathers_qualification_mapping = {
    1: "Secondary Education - 12th Year of Schooling or Equivalent",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    13: "2nd year complementary high school course",
    14: "10th Year of Schooling",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent",
    20: "Complementary High School Course",
    22: "Technical-professional course",
    25: "Complementary High School Course - not concluded",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th year of schooling",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without having a 4th year of schooling",
    37: "Basic education 1st cycle (4th/5th year) or equivalent",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equivalent",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)"
}

mothers_occupation_mapping = {
    0: "Student",
    1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative staff",
    5: "Personal Services, Security and Safety Workers and Sellers",
    6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    7: "Skilled Workers in Industry, Construction and Craftsmen",
    8: "Installation and Machine Operators and Assembly Workers",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "Not Available",
    122: "Health professionals",
    123: "Teachers",
    125: "Specialists in Information and Communication Technologies (ICT)",
    131: "Intermediate level science and engineering technicians and professions",
    132: "Technicians and professionals, of intermediate level of health",
    134: "Intermediate level technicians from legal, social, sports, cultural and similar services",
    141: "Office workers, secretaries in general and data processing operators",
    143: "Data, accounting, statistical, financial services and registry-related operators",
    144: "Other administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers and the like",
    171: "Skilled construction workers and the like, except electricians",
    173: "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like",
    175: "Workers in food processing, woodworking, clothing and other industries and crafts",
    191: "Cleaning workers",
    192: "Unskilled workers in agriculture, animal production, fisheries and forestry",
    193: "Unskilled workers in extractive industry, construction, manufacturing and transport",
    194: "Meal preparation assistants"
}

fathers_occupation_mapping = {
    0: "Student",
    1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative staff",
    5: "Personal Services, Security and Safety Workers and Sellers",
    6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    7: "Skilled Workers in Industry, Construction and Craftsmen",
    8: "Installation and Machine Operators and Assembly Workers",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "Not Available",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces personnel",
    112: "Directors of administrative and commercial services",
    114: "Hotel, catering, trade and other services directors",
    121: "Specialists in the physical sciences, mathematics, engineering and related techniques",
    122: "Health professionals",
    123: "Teachers",
    124: "Specialists in finance, accounting, administrative organization, public and commercial relations",
    131: "Intermediate level science and engineering technicians and professions",
    132: "Technicians and professionals, of intermediate level of health",
    134: "Intermediate level technicians from legal, social, sports, cultural and similar services",
    135: "Information and communication technology technicians",
    141: "Office workers, secretaries in general and data processing operators",
    143: "Data, accounting, statistical, financial services and registry-related operators",
    144: "Other administrative support staff",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers and the like",
    154: "Protection and security services personnel",
    161: "Market-oriented farmers and skilled agricultural and animal production workers",
    163: "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence",
    171: "Skilled construction workers and the like, except electricians",
    172: "Skilled workers in metallurgy, metalworking and similar",
    174: "Skilled workers in electricity and electronics",
    175: "Workers in food processing, woodworking, clothing and other industries and crafts",
    181: "Fixed plant and machine operators",
    182: "Assembly workers",
    183: "Vehicle drivers and mobile equipment operators",
    192: "Unskilled workers in agriculture, animal production, fisheries and forestry",
    193: "Unskilled workers in extractive industry, construction, manufacturing and transport",
    194: "Meal preparation assistants",
    195: "Street vendors (except food) and street service providers"
}
gender_mapping= {
    0: "Female",
    1: "Male"
}
international_mapping= {
    0: "Not International",
    1: "International"
}

# Define a dictionary that relates column names to their respective mappings
mappings = {
    "Marital status": marital_status_mapping,
    "Application mode": application_mode_mapping,
    "Application order": application_order_mapping,
    "Course": course_mapping,
    "Previous qualification": previous_qualification_mapping,
    "Nacionality": nationality_mapping,
    "Mother's qualification": mothers_qualification_mapping,
    "Father's qualification": fathers_qualification_mapping,
    "Mother's occupation": mothers_occupation_mapping,
    "Father's occupation": fathers_occupation_mapping,
    "Gender": gender_mapping,
    "International": international_mapping,


}



# Apply the mapping to each column using a loop
# Apply the mapping to each column using a loop
for column_name, mapping_dict in mappings.items():
    df_string = cat_to_string(df_VIZ, column_name, mapping_dict)

# Transforming 'Student Status' values into numerical format, making them interpretable by machine learning algorithms
df['Student Status'] = df['Student Status'].map({'Dropout' : 0, 'Enrolled': 1, 'Graduate': 2})

# Removing unnecessary columns that won't contribute to the analysis. dropping values with a corr [-0.05,0.05]
df = df.drop(columns=['Nationality', 'International', 'Educational special needs', 'Course',
                      'Mother\'s qualification','Father\'s qualification',
                      'Mother\'s occupation', 'Father\'s occupation',
                      'Curricular units 1st sem (credited)', 'Curricular units 1st sem (evaluations)',
                      'Unemployment rate', 'Inflation rate', 'GDP'], axis=1)
# Creating interaction features for academic performance
df['Yearly Credit Approved'] = df['Curricular units 1st sem (approved)'] * df['Curricular units 2nd sem (approved)']
df['Yearly Grade'] = df['Curricular units 1st sem (grade)'] * df['Curricular units 2nd sem (grade)']

# Creating aggregated features
df['Total Credit approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
df['Total Grade'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2

# Dropping the original features to reduce multi-collinearity
columns_to_drop = ['Curricular units 1st sem (approved)', 'Curricular units 2nd sem (approved)',
                   'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)',]
df.drop(columns_to_drop, axis=1, inplace=True)

#####################################################################
# TRAINING AND EVALUATION OF THE MODEL

y = df['Student Status']
X = df.drop(['Student Status'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, **kwargs):
    """
    Train a machine learning model and evaluate its performance.

    Parameters:
    - model: The machine learning model to train (e.g., DecisionTreeClassifier()).
    - X_train: Training data features.
    - X_test: Testing data features.
    - y_train: Training data labels.
    - y_test: Testing data labels.
    - **kwargs: Additional keyword arguments to pass to the model's fit method.

    Returns:
    - model: The trained machine learning model.
    - accuracy: The accuracy of the model on the test data.
    - precision: The precision of the model on the test data.
    """
    # Train the model
    model.fit(X_train, y_train, **kwargs)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate f1_score,accuracy and precision
    f1_score = metrics.f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')


    # Print performance metrics
    # st.write(f"Accuracy of {model} is :", accuracy)
    # st.write(f"Precision of {model} is :", precision)

    y_pred = pd.Series(y_pred, index=X_test.index)

    return model,y_pred,f1_score, accuracy, precision



#####################################################################

# Now both experiments are logged to MLflow

#############################

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


# update the metrics based on the model
def update_metrics(model_type, f1_score,accuracy,precision):
    cols = st.columns(3)
    # Check if 'first_run' exists in the session state, if not, initialize it
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
        st.session_state.previous_f1 = 0
        st.session_state.previous_accuracy = 0
        st.session_state.previous_precision = 0

    # Calculate the changes if not the first run
    if st.session_state.first_run:
        f1_change = accuracy_change = precision_change = 0
        st.session_state.first_run = False  # Set first run to False after the first check
    elif st.session_state.previous_precision != 0 and st.session_state.previous_accuracy != 0 and st.session_state.previous_f1 != 0:
        f1_change = round((f1_score - st.session_state.previous_f1) / st.session_state.previous_f1 * 100, 3)
        accuracy_change = round(
            (accuracy - st.session_state.previous_accuracy) / st.session_state.previous_accuracy * 100, 3)
        precision_change = round(
            (precision - st.session_state.previous_precision) / st.session_state.previous_precision * 100, 3)
    else:
        f1_change = accuracy_change = precision_change = 0

    # Update the previous metrics
    st.session_state.previous_f1 = f1_score
    st.session_state.previous_accuracy = accuracy
    st.session_state.previous_precision = precision
    with cols[0]:
        ui.metric_card(title=f"{model_type}' f1-Score",
                       content=f"{round(f1_score,3) *100}%",
                       description=f"{f1_change}% from last run",
                       key="card1")
    with cols[1]:
        ui.metric_card(title="Accuracy",
                       content=f"{round(accuracy,3)*100}%",
                       description=f"{accuracy_change}% from last run",
                       key="card2")
    with cols[2]:
        ui.metric_card(title="Precision",
                       content=f"{round(precision,4)*100}%",
                       description=f"{precision_change}% from last run",
                       key="card3")

if selected == "Overview":
    st.title("Overview")
    st.markdown("""
    ### 🧐Dataset Overview

    Our dataset provides an overview of student demographics, educational paths, and outcomes within the Portuguese education system. It includes a variety of attributes including:
    - **Personal Information:** Age, gender, marital status.
    - **Academic Details:** Course enrollment, previous qualifications, and academic performance across semesters.
    - **Socio-economic Factors:** Parents' occupation and educational levels, scholarship status, and tuition payment statuses.

    Additionally, the dataset integrates broader economic indicators, such as the unemployment rate, inflation rate, and GDP, which may influence student success. However, we will mostly discard these indicators for this study.

    ### 🎯Project Goal

    The goal of the project is to analyze the factors that contribute to educational outcomes such as graduation, retention, and dropout rates among Portuguese students. We aim to identify patterns and correlations that can inform educational policies and intervention strategies to enhance student achievement and retention.
    """)

    looker_link = "https://lookerstudio.google.com/reporting/6141ce7c-954d-4801-bad7-b58131aa563d/page/J1lxD"
    column1, column2, column3 = st.columns([1, 1, 1])
    with column1:
        st.write("")
    with column2:
        ui.link_button(text="👉🏻 Go To Looker Studio", url=looker_link, key="link_btn")
    with column3:
        st.write("")
if selected == "Visualisation":

    tab1, tab2, tab3,tab4 = st.tabs(["Barcharts", "Stacked", "Sankey","Explainable AI"])

    with tab1:
        st.subheader("Percentage of Output by Gender")
        # Group the output based on the gender and count how many there is in each category
        # create an extra column in our new dataframe called counts
        df_counts = df_VIZ.groupby(['Gender', 'Output']).size().reset_index(name='Count')

        # find the total number of
        total_counts = df_counts.groupby('Gender')['Count'].transform('sum')

        # Calculate percentage
        df_counts['Percentage'] = 100 * df_counts['Count'] / total_counts

        # Plot configuration
        plt.figure(figsize=(12, 8))
        plt.title('Percentage of Output by Gender')

        # Using a bar plot to show the percentages of 'Output' values for each 'Gender'
        sns.barplot(data=df_counts, x='Gender', y='Percentage', hue='Output', palette='pastel', dodge=True)

        # Adjust legend
        plt.legend(title='Output')

        # Show plot
        st.pyplot()

        paragraphs = [
            "Graduation Rate:",
            "A smaller proportion of female students graduate compared to their male counterparts,as indicated by the green bars. Females show approximately a 60% graduation rate, while males reach almost 40%.",
            "Dropout Rate:",
            "The dropout rate for female students is significantly lower than for males, with about 20% of females dropping out.",
            "The dropout rate for males is lower than their graduation rate but still substantial, roughly around 30%."]
        for paragraph in paragraphs:
            st.write(paragraph)
            # Filter the DataFrame to include rows with specified marital status values
        filtered_df = df_VIZ[df_VIZ['Marital status'].isin(['divorced', 'married', 'single'])]

        df_counts_marital_status = filtered_df.groupby(['Marital status', 'Output']).size().reset_index(name='count')

        # Plot configuration
        plt.figure(figsize=(12, 8))
        plt.title('Count of Output by Marital Status')

        # Using barplot to show the counts of 'Output' values for each 'Marital status'
        sns.barplot(data=df_counts_marital_status, x='Marital status', y='count', hue='Output', palette='pastel')

        # Adjust legend
        plt.legend(title='Output')

        # Show plot
        st.pyplot()
    with tab2:
        st.subheader("Impact of Mother's Occupation on Student Outcomes")

        # Filter rows where "Mother's occupation" is not numeric
        filtered_df = df_VIZ[~df_VIZ["Mother's occupation"].astype(str).str.isnumeric()]

        # Group the filtered data by "Mother's occupation" and "Output"
        grouped_data = filtered_df.groupby(["Mother's occupation", 'Output']).size().unstack(fill_value=0)

        # Reset index to make "Mother's occupation" a column again for easier plotting
        grouped_data.reset_index(inplace=True)

        # Plotting
        plt.figure(figsize=(14, 8))

        # Plotting each category as a separate bar with appropriate stacking
        sns.barplot(x="Mother's occupation", y="Graduate", data=grouped_data, color="green", label="Graduate")
        sns.barplot(x="Mother's occupation", y="Dropout", data=grouped_data, color="red", label="Dropout",
                    bottom=grouped_data["Graduate"])
        sns.barplot(x="Mother's occupation", y="Enrolled", data=grouped_data, color="blue", label="Enrolled",
                    bottom=grouped_data["Graduate"] + grouped_data["Dropout"])

        # Customize plot appearance
        plt.xticks(rotation=90)
        plt.xlabel("Mother's Occupation")
        plt.ylabel("Number of Students")
        plt.title("Impact of Mother's Occupation on Student Outcomes")
        plt.legend(title="Output")
        plt.tight_layout()
        st.pyplot()

        st.subheader("Impact of Father's Occupation on Student Outcomes")
        # Grouping the data by "Father's occupation" and "Output"
        # But since of the occupations were numeric I first drop them so that we can just look at the ones that are strings

        # Filter rows where "Mother's occupation" is not numeric
        filtered_df = df_VIZ[~df_VIZ["Father's occupation"].astype(str).str.isnumeric()]

        # Group the filtered data by "Father's occupation" and "Output"
        grouped_data = filtered_df.groupby(["Father's occupation", 'Output']).size().unstack(fill_value=0)

        # Reset index to make "Father's occupation" a column again for easier plotting
        grouped_data.reset_index(inplace=True)

        # Plotting
        plt.figure(figsize=(14, 8))
        sns.barplot(x="Father's occupation", y="Graduate", data=grouped_data, color="green", label="Graduate")
        sns.barplot(x="Father's occupation", y="Dropout", data=grouped_data, color="red", label="Dropout",
                    bottom=grouped_data["Graduate"])
        sns.barplot(x="Father's occupation", y="Enrolled", data=grouped_data, color="blue", label="Enrolled",
                    bottom=grouped_data["Graduate"] + grouped_data["Dropout"])

        plt.xticks(rotation=90)
        plt.xlabel("Father's Occupation")
        plt.ylabel("Number of Students")
        plt.title("Impact of Father's Occupation on Student Outcomes")
        plt.legend(title="Output")
        plt.tight_layout()
        st.pyplot()

        st.write(
            "The graphs above illustrate the impact of parental occupation on student outcomes, categorized by 'Graduate', 'Dropout', and 'Enrolled' statuses.")
        paragraphs = [
            "Both graphs show that parents in more stable and intellectually-oriented professions (Administration, Armed forces) tend to have children who graduate at higher rates. This might be due to both economic stability and a cultural emphasis on the value of education in these families.",
            "In both cases, occupations with lower socio-economic status correlate with higher dropout rates. This could indicate financial pressures or less available time with parents, which impacts educational support.",
            "We can also observe that for some domains, the impact of fathers' occupations on dropout rates is more pronounced compared to mothers' occupations, possibly reflecting traditional gender roles where fathers' income and job stability might weigh more heavily on family decisions."]
        for paragraph in paragraphs:
            st.markdown(paragraph)
    with tab3:
        st.subheader("Student Pathways - Sankey Plot")

        # Mapping labels for evening attendance and output
        evening_label = {0: 'Day Classes', 1: 'Evening Classes'}
        output_label = {'Graduate': 'Graduated', 'Dropout': 'Dropped Out', 'Enrolled': 'Enrolled in School'}

        # Apply mappings to update DataFrame
        df_updated = df_VIZ.copy()
        df_updated['evening attendance'] = df_updated['evening attendance'].map(evening_label)
        df_updated['Output'] = df_updated['Output'].map(output_label)

        # Create a summary DataFrame for Sankey plot
        summary_df = df_updated.groupby(['Output', 'evening attendance']).size().reset_index(name='Count')

        # Define unique labels for nodes and colors
        label_list = list(set(summary_df['evening attendance']).union(set(summary_df['Output'])))
        color_map = {'Day Classes': 'lightgreen', 'Evening Classes': 'mediumseagreen',
                     'Graduated': 'lightcoral', 'Dropped Out': 'indianred', 'Enrolled in School': 'goldenrod'}
        node_colors = [color_map[label] for label in label_list]

        # Create lists for source, target, and value
        source, target, value = [], [], []
        for index, row in summary_df.iterrows():
            source.append(label_list.index(row['evening attendance']))
            target.append(label_list.index(row['Output']))
            value.append(row['Count'])

        # Define link colors based on source or target
        link_colors = [color_map[label_list[source[i]]] for i in range(len(source))]

        # Create Sankey diagram figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=label_list, color=node_colors),
            link=dict(source=source, target=target, value=value, hoverinfo='all', color=link_colors)
        )])

        # Update layout for the Sankey plot
        fig.update_layout(title_text="Student Pathways", font_size=10)

        # Display the Sankey diagram within Streamlit
        st.plotly_chart(fig)
        paragraphs = [
            "Graduation Rates:",
            "Evening classes show a higher graduation rate than day classes. This could be because students who take evening classes are often working individuals who are more determined to finish their education quickly due to career commitments.",
            "Dropout Rates:",
            "Both class schedules show dropouts, but the rate is less pronounced for day classes. This might indicate that students in day classes have more flexible schedules or fewer outside commitments, reducing pressure and the likelihood of dropping out."]
        for paragraph in paragraphs:
            st.markdown(paragraph)

        st.subheader("Student Pathways -Parallel Plot")
        # Map your values and create a new DataFrame for parallel plot
        df_parallel = df_VIZ.copy()
        df_parallel['Tuition fees up to date'] = df_parallel['Tuition fees up to date'].map(
            {0: 'Not up to date', 1: 'Up to date'})
        df_parallel['Output'] = df_parallel['Output'].map(
            {'Graduate': 'Graduated', 'Dropout': 'Dropped Out', 'Enrolled': 'Enrolled in School'})
        df_parallel['Scholarship holder'] = df_parallel['Scholarship holder'].map({0: 'No', 1: 'Yes'})

        # Assign colors based on 'Scholarship holder'
        color_map = {'No': 'blue', 'Yes': 'orange'}
        df_parallel['color'] = df_parallel['Scholarship holder'].map(color_map)

        # Create Parcats plot using Plotly
        fig = go.Figure(data=
        go.Parcats(
            dimensions=[
                {'label': 'Scholarship', 'values': df_parallel['Scholarship holder']},
                {'label': 'Tuition Status', 'values': df_parallel['Tuition fees up to date']},
                {'label': 'Output', 'values': df_parallel['Output']}
            ],
            line={'color': df_parallel['color'], 'colorscale': 'Viridis'},  # Color lines by scholarship status
        )
        )

        # Update layout
        fig.update_layout(title="Student Pathways", width=800)

        # Display the Parcats plot within Streamlit
        st.plotly_chart(fig)
        paragraphs = [
            "A significant flow from students with scholarships maintains tuition payments up to date, which likely supports their ability to continue education and possibly graduate.",
            "The transitions from having a scholarship and keeping tuition up to date towards graduation appear strong, suggesting that scholarships might help students successfully complete their courses.",
            "There is a smaller but notable flow towards students dropping out or staying enrolled, even with scholarships, indicating that while financial support helps, it may not be sufficient to guarantee graduation for all students."]
        for paragraph in paragraphs:
            st.markdown(paragraph)
    with tab4:
        st.markdown('<center><h2>Explainable AI</h2></center>', unsafe_allow_html=True)
        st.write(""" Shapash is User-friendly Explainability and Interpretability app that helps  Develop Reliable and Transparent Machine Learning Models 
        in this case it will help us see and understand which variables have the most impact and contribute more towards our model prediction. We chose a couple a graphs that seemed to be the most helpfull to our case""")
        # Assuming images are in the same directory as the script

        st.image(img_importance)
        st.write("""The feature importance plot shows the most important features in the dataset. The importance of a feature is calculated based on the contribution of the feature to the model's predictions. The higher the importance, the more the feature contributes to the model's predictions.
        Here we have 5 to 7 variables that are very important. with The Yearly Approved Credit contributing the most.
        """)

        st.image(img_importance_subset)
        st.write("Same as the previous graph but with a subset of the most important features.")

        st.image(img_contribution_subset)
        st.write("""The feature contribution plot shows the contribution of each feature to the model's predictions for each individual prediction. The contribution of a feature is calculated based on the feature's impact on the model's prediction for a specific instance. The higher the contribution, the more the feature influences the model's prediction for that instance.
        Here we can see that for the yearly credit approved, the more credits taken then higher the chances of the student not dropping out.""")

if selected == "Prediction":
    menu2 = option_menu(
        menu_title=None,
        options=["Models", "ML Flow"],
        icons=["bookmark", "activity"],
        default_index=0,
        orientation="horizontal",

    )
    if menu2 == "Models":
        prediction_type = st.sidebar.selectbox('Select Type of Prediction', ['Decision Tree (Default)', 'KNN'])

        if prediction_type == "Decision Tree (Default)":
            st.title("Decision Tree Prediction")
            max = st.number_input("Enter the maximum depth of the decision tree (5 is the best)", 1, 10, value = 1, placeholder= "Enter a number")
            decision_tree_model, y_pred,dt_f1_score, dt_accuracy, dt_precision = train_and_evaluate_model(
                DecisionTreeClassifier(max_depth=max),
                X_train, X_test, y_train, y_test
            )
            update_metrics("Decision Tree", dt_f1_score, dt_precision, dt_accuracy)
            # Export the tree in Graphviz format
            feature_names = X.columns
            feature_cols = X.columns
            dot_data = export_graphviz(decision_tree_model, out_file=None,
                                       feature_names=feature_cols,
                                       class_names=["0", "1", "2"],
                                       filled=True, rounded=True,
                                       special_characters=True)

            # Convert to a graph using Graphviz
            graph = graphviz.Source(dot_data)


            # Function to display Graphviz tree in Streamlit
            def st_graphviz(graph, width= None, height=None):
                graphviz_html = f"<body>{graph.pipe(format='svg').decode('utf-8', errors='replace')}</body>"
                st.components.v1.html(graphviz_html,width = width , height=height, scrolling=True)


            # Display the tree in Streamlit
            st.title('Decision Tree Visualization')
            st_graphviz(graph,1200, 800)

            st.markdown("""
            ### Path Description:
    
            **Starting Point (Root Node):**  
            The root node is the most significant on the prediction tree. It is the first decision point where the tree splits into branches based on the student's yearly credit approval.
            The question we can ask is: "Is the student's yearly credit more than 15.5 or less?" And depending on the answer, we move down the tree to the next question.
    
            **First Decision - True (Yes, 15.5 or less):**  
            For Yes, we can move to the next question down the left branch of the tree.
    
            **Second Question:**  
            The next question is: "Is the student's yearly credit approved 4.5 or less?"  
            This further refines our group of students, focusing on those who have very few credits for the year.
    
            **Second Decision - True (Yes, 4.5 or less):**  
            We again answer yes and proceed to a final category in this path for this example depth 2.
    
            **Outcome (Leaf Node):**  
            The leaf node we reach after these two "yes" answers shows:  
            - **Gini:** 0.327 (This is a measure of uncertainty or impurity. The lower the value the better for the uniformity of the groups.
            A lower value, like 0.327, suggests that the node is more or less pure, suggesting that most students in this node fall into the same category.)  
            - **Samples:** 733 (This is the number of students who fit this profile.)  
            - **Values:** [593, 84, 56] (This tells us how many students are predicted to dropout, stay enrolled, or graduate. Here, 593 are predicted to dropout, 84 to stay enrolled, and 56 to graduate.)  
            - **Majority Class:** 0 (Most students in this group, those with very low credit approval, are predicted to dropout.)
            """)

        elif prediction_type == "KNN":
            st.title("KNN Prediction")
            #KNN Classifier
            k_neighbors = st.number_input("Enter the number of neighbors for the KNN model",1,100,value = 10, placeholder= "Enter a number")
            knn_model, y_pred, knn_f1_score, knn_accuracy, knn_precision = train_and_evaluate_model(
                KNeighborsClassifier(n_neighbors=int(k_neighbors)),
                X_train, X_test, y_train, y_test)
            update_metrics("KNN", knn_f1_score,knn_accuracy,knn_precision)

            # Scale your data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # User input for the number of neighbors
            # Define the range of k values dynamically based on user input
            max_k = k_neighbors + 20
            k_list = list(range(1, max_k + 1))
            k_values = dict(n_neighbors=k_list)

            # Perform grid search with the list of k values
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=k_values, cv=5, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)

            # Get the results into a DataFrame
            results_df = pd.DataFrame(grid_search.cv_results_)


            results_df = pd.DataFrame(grid_search.cv_results_)

            # Sort the DataFrame by 'mean_test_score' and 'std_test_score' and then take the top 5
            top_results = results_df.sort_values(by=['mean_test_score', 'std_test_score'], ascending=[False, True]).head(5)

            # Display the DataFrame in Streamlit
            st.write("Top 5 K Values by Mean Test Score and Stability:")

            st.dataframe(top_results[['params', 'mean_test_score', 'std_test_score']])
            # Plotting the mean test scores
            graphic = results_df['mean_test_score']
            plt.figure(figsize=(10, 5))
            plt.plot(k_list, graphic, color='navy', linestyle='dashed', marker='o')
            plt.xlabel('K Number of Neighbors', fontdict={'fontsize': 12})
            plt.ylabel('Accuracy', fontdict={'fontsize': 12})
            plt.title('K NUMBER X ACCURACY', fontdict={'fontsize': 24})
            plt.xticks(range(0, max_k, max(1, max_k // 10)))  # Adjust x-ticks dynamically
            st.pyplot(plt)

    if menu2 == "ML Flow":
        st.title("ML FLOW Visualization")
        mlflowlink = "https://dagshub.com/Danjari/Dropout.mlflow/#/compare-experiments/s?experiments=%5B%220%22%2C%221%22%5D&searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D"
        column1, column2, column3 = st.columns([1,1,1])
        with column1:
            st.write("")
        with column2:
            ui.link_button(text="👉🏽 Go To ML Flow", url=mlflowlink, key="link_btnmlflow")
        with column3:
            st.write("")

        #####################################################################
        def main():
            st.markdown("## Model Experimentation with MLflow")

            # File upload
            uploaded_file = st.file_uploader("Choose a file (CSV or Excel)")
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Validate data
                    if not all(df.dtypes.apply(
                            lambda dtype: pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(
                                dtype))):
                        st.error("All columns must be numeric (float or int). Please upload a cleaned dataset.")
                        st.stop()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    st.stop()
            else:
                st.stop()

            # Problem type selection
            problem_type = st.selectbox("Select the problem type", ["classification", "regression"])

            # Model selection based on problem type
            MODELS = {
                "classification": {
                    "KNN": KNeighborsClassifier,
                    "Decision Tree": DecisionTreeClassifier,
                    "Logistic Regression": LogisticRegression
                },
                "regression": {
                    "LR": LinearRegression,

                }
            }

            model_options = list(MODELS[problem_type].keys())
            model_choice = st.selectbox("Choose a model", model_options)

            # Feature and target selection
            if len(df.columns) > 1:
                target = st.selectbox("Select the target variable", df.columns)
                feature_options = [col for col in df.columns if col != target]
                features = st.multiselect("Choose some features", feature_options, default=feature_options)
            else:
                st.error("Dataset must contain more than one column.")
                st.stop()

            # MLflow tracking
            track_with_mlflow = st.checkbox("Track with mlflow?")

            # Model training
            start_training = st.button("Start training")
            if start_training:
                if track_with_mlflow:
                    mlflow.set_experiment("User_Uploaded_Data")
                    with mlflow.start_run():
                        train_and_evaluate(df, features, target, model_choice, problem_type, MODELS,
                                           track_with_mlflow)

        def train_and_evaluate(df, features, target, model_choice, problem_type, MODELS, track_with_mlflow):
            X = df[features].copy()
            y = df[target].copy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = MODELS[problem_type][model_choice]()
            model.fit(X_train, y_train)

            # Model evaluation
            preds_train = model.predict(X_train)
            preds_test = model.predict(X_test)
            if problem_type == "classification":
                metric_train = f1_score(y_train, preds_train, average='micro')
                metric_test = f1_score(y_test, preds_test, average='micro')
                metric_name = "f1_score"
                
            else:
                metric_train = r2_score(y_train, preds_train)
                metric_test = r2_score(y_test, preds_test)
                metric_name = "r2_score"

            st.write(f"{metric_name}_train", round(metric_train, 3))
            st.write(f"{metric_name}_test", round(metric_test, 3))

            if track_with_mlflow:
                mlflow.log_param('model', model_choice)
                mlflow.log_param('features', features)
                mlflow.log_metric(metric_name + "_train", metric_train)
                mlflow.log_metric(metric_name + "_test", metric_test)


        if __name__ == '__main__':
            main()

        #####################################################################

if selected == "Conclusion":
    st.title("Conclusion 🎤")
    st.markdown("""
    **1. Data Quality and Preparation**  
    **Address Missing Values**: Given the socio-economic factors involved in our dataset, it is important to take note of how we handle missing values. It is crucial to use domain knowledge to remove missing values in a way that does not introduce bias.  
    To improve the accuracy of our model, we could also introduce new features that can help in making better predictive decisions.  
    For example, introducing new variables such as "parental job stability," "education policies," etc.  

    **2. Model-related improvements**  
    For a decision tree classifier, it is important to limit the growth of the tree to prevent overfitting but we also have to avoid underfitting.  
    Even though we have a way to calculate the most optimal K-value, we can't be certain that that is the best value for our model. It may be that the 1000th iteration of cross-validation will provide a different optimal value. It is crucial to test and validate different parameters to ensure the model's accuracy and reliability.  

    **3. Long-term:**  
    Since we are dealing with education data, it is important to continuously update the model with new data, such as changes in the economic landscape or educational policies in Portugal, to keep the model relevant and accurate.  
    Additionally, we could also merge our current dataset with other datasets that may provide additional insights. By incorporating external datasets, we can enhance the quality and accuracy of our model predictions.
    """)
    # Stop the emissions tracker
    emissions = tracker.stop()


    st.write(f"Total CO2 emissions:{emissions:.4f}kg CO2")
