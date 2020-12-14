import sqlite3 as sl
import pandas as pd

con = sl.connect('training.db')

df_skill = pd.DataFrame({
'weight': [72, 72, 133, 85, 115, 120, 107, 85, 103, 60, 68, 91, 111, 108, 82, 75, 89, 78, 66, 92, 80, 102, 80, 104, 70, 92, 95, 94, 80, 83, 69, 54, 48, 100, 105, 70, 60, 86, 83, 90, 88, 66, 100, 108, 83, 86, 80, 100, 100, 83],
'height': [170, 168, 175, 185, 173, 188, 160, 180, 177, 161, 168, 180, 182, 187, 174, 170, 168, 175, 164, 183, 174, 167, 170, 167, 164, 176, 172, 178, 170, 172, 164, 165, 164, 180, 180, 174, 164, 168, 178, 180, 176, 169, 175, 170, 172, 178, 181, 178, 176, 178],
'age': [75, 50, 61, 52, 64, 71, 64, 51, 74, 64, 61, 77, 52, 64, 64, 66, 65, 64, 67, 62, 68, 60, 62, 64, 69, 62, 73, 68, 83, 70, 71, 73, 64, 60, 64, 63, 67, 50, 64, 68, 55, 73, 59, 69, 70, 60, 64, 63, 61, 63],
'psa': [131, 69.9, 16.89, 160.6, 46.44, 33.8, 35.1, 6.03, 5.2, 50, 15.12, 7.07, 52, 6.7, 26, 33.5, 36.42, 150, 29.4, 56, 46, 9.54, 35.1, 7.9, 4.6, 73.8, 16.2, 1322, 70, 14.27, 58.8, 76.09, 40.73, 11, 49.4, 44.25, 136, 69.9, 20.49, 756, 21.02, 6.56, 164.9, 160, 14.27, 124, 1129, 39.1, 8.3, 8.2],
'prostate_volume': [85.5, 30, 44, 60, 27, 57, 58, 10, 34, 32, 19, 27, 43, 42, 71, 49, 29, 71, 36, 37, 44, 48, 55, 51, 20, 29, 15, 95, 34, 39, 47, 56, 32, 9, 46, 54, 62, 30, 15, 35, 47, 32, 18, 28, 39, 71, 53, 34, 14, 13]
})

df_skill.to_sql('PATIENT', con)


