# Who Survived the Titanic?

**Minor Project – 2 | Data Analysis and Visualization using Python**
Praveen Yadav, MCA, Lovely Professional University

## About this project

I picked the Titanic passenger data because it is a real dataset that is easy to relate to. The question I wanted to answer is a simple one: did everyone on the ship have the same chance of surviving, or did things like gender, ticket class and age make a difference?

I cleaned the data, explored it, made 8 charts and wrote down what I found. The full write-up is in the `report` folder.

## Dataset

- **Name:** Titanic – Machine Learning from Disaster
- **Source:** Kaggle ([https://www.kaggle.com/c/titanic](https://www.kaggle.com/datasets/yasserh/titanic-dataset))
- **Size:** 891 passengers, 12 columns
- **File:** `data/titanic.csv` (same data as Kaggle's `train.csv`)

## What the code does

1. Loads the data and checks its size and column types.
2. Finds the missing values and fixes them. Age is filled with the median, Embarked with the most common port, and Cabin is dropped because about 77% of it is empty.
3. Makes two new columns: `FamilySize` and `AgeGroup`.
4. Prints survival rates by gender, class, age group and family size.
5. Saves 8 charts (pie, bar, histogram, box plot, line and heatmap) into the `charts` folder.

## How to run it

You need Python 3 installed. Then, from inside this folder:

```
pip install -r requirements.txt
python project.py
```

The numbers are printed in the terminal and the charts appear in the `charts` folder.

If you prefer a notebook, open `titanic-analysis.ipynb` in Jupyter, VS Code or Google Colab. It has the same analysis with a short explanation after each chart, and the results are already saved inside it.

## Folder structure

```
titanic-survival-analysis/
├── README.md
├── requirements.txt
├── project.py                      # the whole analysis in one script
├── Titanic_Analysis.ipynb          # same analysis as a Jupyter notebook
├── data/
│   └── titanic.csv                 # the dataset
├── charts/                         # 8 charts made by project.py
└── report/
    ├── Minor_Project_2_Report.docx # project report (editable)
```

## What I found

- Only **38.4%** of the passengers survived.
- **Women** survived far more often than men (74% vs 19%).
- **First class** did much better than third class (63% vs 24%).
- **Children** had the best chances and seniors the worst.
- **Small families** did best, while people travelling alone or in very large families did worse.
- Gender and class together say the most: a first-class woman had about a 97% chance of surviving, a third-class man only about 14%.

## Things to keep in mind

- This shows patterns, not proof of cause. Class and fare are closely linked, so I can't say which one mattered more.
- The spike at age 28 in the age histogram appears because I filled the missing ages with the median (28).

## Tools used

Python 3, Pandas, Matplotlib, Seaborn
