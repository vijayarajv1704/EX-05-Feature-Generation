# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```
Developed By : Vijayaraj V
Reg No: 212222230174
```
```py
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
```py
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4

```
```py
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT
## data.csv :
### Data:
![Dataset](https://user-images.githubusercontent.com/94525786/234182599-3822b87a-99d6-456f-bdfa-f933cbe639e8.png)

### Ordinary Encoder-[col-[ord_1, ord_2]:
![Ordinary encoder2](https://user-images.githubusercontent.com/94525786/234182637-7a07e023-1253-47e5-b4e1-cfdddec33a7a.png)

### Binary Encoding-[Columns-bin_1, bin_2]:
![Binary Encoder](https://user-images.githubusercontent.com/94525786/234182582-fce3462d-dc0e-4a9e-bcb9-16abf6391431.png)

### Encoded Data:
![Encoded data](https://user-images.githubusercontent.com/94525786/234182614-a0509e9a-f024-4fa7-911f-e7f865f5eda0.png)

### Standard Scalar
![StandardScalar](https://user-images.githubusercontent.com/94525786/234182657-c21335c8-67e3-4bc2-be1a-8327a6feb9d7.png)

### MaxAbs Scalar:  
![MaxAbsscalar](https://user-images.githubusercontent.com/94525786/234182619-7acc269f-40f1-42a3-88d4-de914a66ef5e.png)

### MinMax Scalar:
![Minmax Scalar](https://user-images.githubusercontent.com/94525786/234182628-56cc90b3-68b2-453d-8081-471d963ff588.png)

### Robust Scalar:
![RobustScalar](https://user-images.githubusercontent.com/94525786/234182647-56f5e69b-9eff-4666-a94f-5f86b55c6014.png)

## Encoding Data.Csv:

### Data:
![Dataset1](https://user-images.githubusercontent.com/94525786/234186481-10e38d98-1ca0-451f-bf4a-2cb5c0db0159.png)

### Ordinary Encoder :
![Ordinary encoder1](https://user-images.githubusercontent.com/94525786/234186560-0d4fc523-c871-4d78-98de-6980ba8f1914.png)

### Binary Encoding :
![Binary Encoder1](https://user-images.githubusercontent.com/94525786/234186602-56d494d7-d80a-488c-b180-60bb34246cea.png)

### One Hot Encoded Data:

![One Hot Encoding1](https://user-images.githubusercontent.com/94525786/234186843-695f45f9-d3d2-404b-a79e-98b9e8c01a90.png)

### Standard Scalar:

![StandardScalar1](https://user-images.githubusercontent.com/94525786/234186873-4552304a-c6b6-4686-a0a9-f05e6fdc9f06.png)

### MaxAbs Scalar:  
![MaxabsScalar1](https://user-images.githubusercontent.com/94525786/234186919-57904cc2-109e-4bb5-ba5f-280a7fc9ff3b.png)


### MinMax Scalar:
![MinmaxScalar1](https://user-images.githubusercontent.com/94525786/234187000-cb7b7ff5-4fa2-4614-bffe-38ddef0bbaf7.png)


### Robust Scalar:
![RobustScalar1](https://user-images.githubusercontent.com/94525786/234187045-d496a9da-954c-4a9c-8b95-796003a44398.png)

## Titanic.CSV :
### Data:
![Dataset2](https://user-images.githubusercontent.com/94525786/234296658-28c45104-e2a1-4a6c-80cf-f4799383ff32.png)


### Ordinary Encoder :
![Ordinary encoder2](https://user-images.githubusercontent.com/94525786/234296762-e949b7d1-2193-40ed-b5e0-97daa4121f8b.png)


### Binary Encoding :
![Binary2](https://user-images.githubusercontent.com/94525786/234296868-390cf580-3774-43f0-b604-89d358b6e32b.png)


### One Hot Encoded Data:

![One Hot Encoding1](https://user-images.githubusercontent.com/94525786/234296988-982682b7-95b6-4414-a836-c99949617d02.png)


### Standard Scalar:

![Standard Scalar](https://user-images.githubusercontent.com/94525786/234297098-64477d76-957a-4c90-9af7-99546047fc4f.png)


### MaxAbs Scalar:  
![Max Abs2](https://user-images.githubusercontent.com/94525786/234297803-a9f22371-7339-4ba7-92ad-032c3e3e479e.png)



### MinMax Scalar:
![MinMaxScalar2](https://user-images.githubusercontent.com/94525786/234297324-82aa0a5a-aa37-44e6-9334-1a89747e4feb.png)

### Robust Scalar:
![Robust Scalar2](https://user-images.githubusercontent.com/94525786/234445669-558cea28-7873-4568-94b0-f9b353bb3c5a.png)


## RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
