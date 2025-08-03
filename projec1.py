
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class MGDR:
    def __init__(self, lr, epoch):
        self.lr = lr
        self.epoch = epoch
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X_train, y_train):
        self.coef_ = np.ones(X_train.shape[1])
        self.intercept_ = 0
        for _ in range(self.epoch):
            for _ in range(X_train.shape[0]):
                idx = np.random.randint(0, X_train.shape[0])
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_
                error = y_train[idx] - y_hat
                intercept_der = -2 * error
                coef_der = -2 * error * X_train[idx]
                self.intercept_ -= self.lr * intercept_der
                self.coef_ -= self.lr * coef_der

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


df = pd.read_csv("/Users/subhasisjena/machine learnning related/placement_data_500.csv")

X = df[["IQ", "CGPA", "Faculty_Rating", "JEE_Rank", "Codeforces_Rating"]].values
y = df["Placement_Score"].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42)


model = MGDR(lr=0.01, epoch=50)
model.fit(X_train, y_train)

iq = int(input("Enter IQ: "))
cgpa = float(input("Enter CGPA: "))
faculty_rating = float(input("Enter Faculty Rating (1-10): "))
jee_rank = int(input("Enter JEE Rank: "))
codeforces_rating = int(input("Enter Codeforces Rating: "))

input_features = [[iq, cgpa, faculty_rating, jee_rank, codeforces_rating]]
input_scaled = scaler.transform(input_features)

predicted_score = np.clip(model.predict(input_scaled)[0], 0, 100)
print(f"\n📊 Predicted Placement Score: {predicted_score:.2f} out of 100")
