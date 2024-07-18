from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st
import pandas as pd
import numpy as np

# Sidebar
st.sidebar.title("Imputation Parameters")
st.sidebar.markdown(
    """
    Use this sidebar selector to dynamically change 
    the imputation values for the data by selecting the imputation method.
    """
)

imputers = ["mean", "median", "most_frequent"]
imputer_selector = st.sidebar.selectbox("Select the imputation method:", imputers)

# Load data
data = pd.read_csv("./data/financials.csv")

# Intro section
st.title("Data Exploration and Preprocessing")
st.write(
    """
    Welcome to this simple example demonstrating some preprocessing steps 
    before modeling the data. These processes involve transformations, 
    cleaning, and filling missing data to prepare a dataset for modeling.
    """
)

# Data loading
st.header("Loading Data")
st.markdown(
    """
    We load the original data to be used. These data can be found in the 
    `./data/financials.csv` file.

    The data can be loaded using pandas with the following lines:
    ```python
    import pandas as pd

    data = pd.read_csv('./data/financials.csv')
    ```

    > **Note:** To display the data on screen, you can directly call the variable
    > from a cell if you use Jupyter Notebooks/Lab or Google Colab or print the data with
    > the `print` function.
    > ```python
    > # From a notebook
    > data
    > # From a script
    > print(data)
    > ```

    When displaying the data, you should see a table like the following:
    """
)
st.dataframe(data)
st.markdown(
    """
    After displaying, we can identify dependent and independent variables. 
    Note that we will consider `COUNTRY`, `AGE`, and `SALARY` as independent 
    variables used to predict the dependent variable `PURCHASE`.
    """
)

# Data imputation
st.header("Data Imputation and Encoding")
st.markdown(
    """
    As mentioned earlier, from the data table displayed, we can observe 
    certain characteristics, like some empty fields (`NaN`s), or categorical variables. 
    We'll fix these issues with data imputation and encoding processes.
    """
)

st.subheader("Data Imputation")
st.markdown(
    """
    For data imputation, we can redefine the necessary fields (that are empty) 
    by adding a numerical value; for this, there are different strategies, 
    as values can be defined by the mean, median, mode, or a constant.

    This imputation process can be performed directly in Python using the 
    `sklearn` package as follows:

    ```python
    from sklearn.impute import SimpleImputer
    import numpy as np

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(data.iloc[:, 1:3])
    data.iloc[:, 1:3] = imputer.transform(data.iloc[:, 1:3])
    ```

    For the `strategy` parameter, you can select another method from the left sidebar
    and compare the different results. You will appreciate the changes directly 
    on the data table in real time. The `strategy` parameter of your code in the 
    `SimpleImputer` object can take values `mean`, `median`, and `most_frequent` 
    (or even a constant).

    > **Note:** I've set the left sidebar so you can continue modifying the imputer 
    > values later and see how the data varies even in scaling.
    """
)

imputer = SimpleImputer(missing_values=np.nan, strategy=imputer_selector)
imputer = imputer.fit(data.iloc[:, 1:3])
data.iloc[:, 1:3] = imputer.transform(data.iloc[:, 1:3])
st.dataframe(data)

st.subheader("Data Encoding")
st.markdown(
    """
    For encoding or converting categorical values to numerical values, 
    we can use several approaches. For the `COUNTRY` values, the first 
    step will be to convert label values to numeric identifiers (integers), 
    and then use _one-hot encoding_ to convert them to unit vectors 
    (representing the element by position). To preserve the original data 
    and see how numerical categories are assigned by country name, I'll create 
    a new column named `LABEL_ENCODING` and reorder it to appear next to the 
    `COUNTRY` column.

    This encoding process can be performed directly in Python using the 
    `sklearn` package as follows:

    ```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    data['LABEL_ENCODING'] = label_encoder.fit_transform(data.iloc[:, 0])
    data = data[['COUNTRY', 'LABEL_ENCODING', 'AGE', 'SALARY', 'PURCHASE']]
    ```

    Note in the code that a new column has been added and reordered to 
    appear alongside the `COUNTRY` column.
    """
)
label_encoder = LabelEncoder()
data["LABEL_ENCODING"] = label_encoder.fit_transform(data.iloc[:, 0])
data = data[["COUNTRY", "LABEL_ENCODING", "AGE", "SALARY", "PURCHASE"]]
st.dataframe(data)

st.markdown(
    """
    We can observe how the values have been assigned; however, as mentioned 
    in the session, it is now necessary to transform these values into representations 
    that have an equitable weight. For this, we will use _one-hot encoding_.

    This encoding process can be performed directly in Python using the 
    `sklearn` package as follows:

    ```python
    from sklearn.preprocessing import OneHotEncoder

    onehot_encoder = OneHotEncoder()
    onehot = onehot_encoder.fit_transform(data[['LABEL_ENCODING']]).toarray()
    ```

    Once we have the transformed data, we can see the result of the 
    transformation, where column `0` corresponds to the value of Germany, 
    column `1` to Spain, and `2` to France.
    """
)
onehot_encoder = OneHotEncoder()
onehot = onehot_encoder.fit_transform(data[["LABEL_ENCODING"]]).toarray()
st.dataframe(onehot)

st.markdown(
    """
    This last matrix of values can be integrated into the original data table, 
    to obtain the final table of independent variables that we can use to 
    create a machine learning model. Only the dependent variable (to be predicted) 
    and the data scaling remain to be transformed.

    Also, we reorder the columns for better control and visual identification 
    of the variables.

    ```python
    data['GERMANY'] = onehot[:, 0]
    data['SPAIN'] = onehot[:, 1]
    data['FRANCE'] = onehot[:, 2]
    data = data[['COUNTRY', 'LABEL_ENCODING', 'GERMANY', 'SPAIN', 'FRANCE', 'AGE', 'SALARY', 'PURCHASE']]
    ```
    """
)
data["GERMANY"] = onehot[:, 0]
data["SPAIN"] = onehot[:, 1]
data["FRANCE"] = onehot[:, 2]
data = data[
    [
        "COUNTRY",
        "LABEL_ENCODING",
        "GERMANY",
        "SPAIN",
        "FRANCE",
        "AGE",
        "SALARY",
        "PURCHASE",
    ]
]
st.dataframe(data)

st.subheader("Dependent Variable")
st.markdown(
    """
    For the dependent variable, we'll repeat the label transformation process.

    ```python
    dep_label_encoder = LabelEncoder()
    data['PURCHASE'] = dep_label_encoder.fit_transform(data.iloc[:, 3])
    ```

    This will change the values of the `PURCHASE` column from `Yes`/`No` to `1`/`0`, 
    as can be seen in the updated data table.
    """
)
dep_label_encoder = LabelEncoder()
data["PURCHASE"] = dep_label_encoder.fit_transform(data.iloc[:, 3])
st.dataframe(data)

st.header("Data Scaling")
st.markdown(
    """
    At this point, we are close to concluding the data cleaning and transformation 
    process, as we only need to scale the independent variables through a normalization 
    or standardization process (using the mean and variance to transform the data 
    to have a normal distribution, hence the term _normalize_).

    This can be done simply, again, using `sklearn`.

    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    data.iloc[:, 2:7] = scaler.fit_transform(data.iloc[:, 2:7])
    ```

    And this way we obtain the final, clean, and transformed data.
    """
)
scaler = StandardScaler()
data.iloc[:, 2:7] = scaler.fit_transform(data.iloc[:, 2:7])
st.dataframe(data)

st.markdown(
    """
    If you want to extract only the independent and dependent variables for use in a model, 
    this can be done very simply by selecting the columns we want for each case.

    ```python
    x = data[['GERMANY', 'SPAIN', 'FRANCE', 'AGE', 'SALARY']].values
    y = data[['PURCHASE']].values
    ```

    Thus, we can see that we have in a variable `x` the matrix with independent data:
    """
)
x = data[["GERMANY", "SPAIN", "FRANCE", "AGE", "SALARY"]].values
st.dataframe(x)

st.markdown(
    """
    And in the variable `y` (dependent) as the list with the variable to be predicted in the model.
    """
)
y = data[["PURCHASE"]].values
st.dataframe(y)

# Data Modeling
st.header("Data Modeling")
st.markdown(
    """
    Data can be modeled simply using a Support Vector Machine (SVM) with the following code:

    ```python
    from sklearn.svm import SVC

    classifier = SVC(kernel='rbf')
    classifier.fit(x, y)
    ```

    Once the model is trained, you can predict new values.

    ```python
    classifier.predict([[1.5275, -0.6547, -0.8165, 1.6308, 1.7521]])
    # Output: array(['No'], dtype=object)
    ```
    """
)
