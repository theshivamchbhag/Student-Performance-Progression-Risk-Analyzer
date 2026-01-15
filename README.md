import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("StudentsPerformance.csv")

print("First 5 records of dataset:\n")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nStatistical summary of scores:")
print(df[['math score', 'reading score', 'writing score']].describe())

df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
df['score_std'] = df[['math score', 'reading score', 'writing score']].std(axis=1)

def classify_student(avg_score):
    if avg_score >= 75:
        return 'Good'
    elif avg_score >= 50:
        return 'Average'
    else:
        return 'At Risk'

df['performance_category'] = df['average_score'].apply(classify_student)

print("\nPerformance Category Count:")
print(df['performance_category'].value_counts())

plt.figure()
plt.hist(df['average_score'], bins=10)
plt.xlabel('Average Score')
plt.ylabel('Number of Students')
plt.title('Distribution of Average Student Scores')
plt.show()

category_counts = df['performance_category'].value_counts()

plt.figure()
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
plt.title('Student Performance Categories')
plt.show()

subject_means = df[['math score', 'reading score', 'writing score']].mean()

plt.figure()
plt.bar(subject_means.index, subject_means.values)
plt.xlabel('Subjects')
plt.ylabel('Average Score')
plt.title('Subject-wise Average Scores')
plt.show()

gender_avg = df.groupby('gender')['average_score'].mean()

plt.figure()
plt.bar(gender_avg.index, gender_avg.values)
plt.xlabel('Gender')
plt.ylabel('Average Score')
plt.title('Gender-wise Average Performance')
plt.show()

print("\nProject Analysis Completed Successfully.")
print("This project identifies student performance levels and academic risk using simple statistical analysis.")

