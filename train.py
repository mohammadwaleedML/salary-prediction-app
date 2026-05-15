import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import joblib
BASE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

df = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "salary_dataset_500.csv")
)


# print(df.head())


df["age"]= df["age"].fillna(df["age"].mean()).astype(int)
df["experience"]=df["experience"].fillna(df["experience"].mean()).astype(int)
# print(df.isnull().sum())

df["gender"]=df["gender"].fillna(df["gender"].mode()[0])
df["education"]= df["education"].fillna(df["education"].mode()[0])
df["skills"]=df["skills"].fillna(df["skills"].mode()[0])
df["city"]=df["city"].fillna(df["city"].mode()[0])
df["company_size"]=df["company_size"].fillna(df["company_size"].mode()[0])
# print(df.isnull().sum())

# lable encoding

gender_encoder = LabelEncoder()
education_encoder = LabelEncoder()
skills_encoder = LabelEncoder()
city_encoder = LabelEncoder()
company_encoder = LabelEncoder()

df["gender"] = gender_encoder.fit_transform(df["gender"])
df["education"] = education_encoder.fit_transform(df["education"])
df["skills"] = skills_encoder.fit_transform(df["skills"])
df["city"] = city_encoder.fit_transform(df["city"])
df["company_size"] = company_encoder.fit_transform(df["company_size"])



# le = LabelEncoder()

# df["gender"] = le.fit_transform(df["gender"])
# df["education"] = le.fit_transform(df["education"])
# df["skills"] = le.fit_transform(df["skills"])
# df["city"] = le.fit_transform(df["city"])
# df["company_size"] = le.fit_transform(df["company_size"])

# print(df.head())


x = df.drop("salary", axis=1)
y = df["salary"]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train[["age", "experience"]] = scaler.fit_transform(
    X_train[["age", "experience"]]
)

X_test[["age", "experience"]] = scaler.transform(
    X_test[["age", "experience"]]
)

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

cvs = cross_val_score(model,x,y,cv=5)


print("Cross Validation Score : ",cvs.mean())
print("Accurancy :",r2_score(y_test,prediction))
print("Mean Squared error :",mean_squared_error(y_test,prediction))
print("Mean absolute error :",mean_absolute_error(y_test,prediction))






joblib.dump(model,os.path.join(BASE_DIR, "model", "salary_prediction_model.pkl"))

joblib.dump(scaler,os.path.join(BASE_DIR, "model", "scaler.pkl"))

joblib.dump(gender_encoder,os.path.join(BASE_DIR, "model", "gender_encoder.pkl"))

joblib.dump(education_encoder,os.path.join(BASE_DIR, "model", "education_encoder.pkl"))

joblib.dump(skills_encoder,os.path.join(BASE_DIR, "model", "skills_encoder.pkl"))

joblib.dump(city_encoder,os.path.join(BASE_DIR, "model", "city_encoder.pkl"))

joblib.dump(company_encoder,os.path.join(BASE_DIR, "model", "company_encoder.pkl"))

print("\nModel and encoders saved successfully!")
