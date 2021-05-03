import sqlite3 as sl
import pandas as pd

con = sl.connect('prostate_cancer.db')

df_skill = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243],
    'age': [58, 61, 76, 70, 68, 49, 59, 60, 67, 61, 52, 73, 63, 68, 64, 61, 72, 77, 62, 67, 77, 66, 56, 64, 76, 67, 64, 54, 59, 64, 59, 67, 70, 62, 57, 75, 61, 70, 64, 59, 72, 51, 70, 50, 70, 64, 81, 64, 64, 59, 70, 59, 56, 61, 72, 61, 62, 52, 73, 61, 75, 61, 70, 77, 71, 53, 69, 83, 63, 69, 68, 52, 64, 57, 64, 62, 76, 77, 55, 55, 74, 55, 54, 69, 68, 60, 67, 73, 69, 61, 69, 60, 67, 73, 75, 61, 68, 60, 62, 61, 69, 54, 69, 77, 71, 63, 58, 60, 59, 64, 65, 44, 60, 61, 62, 71, 66, 62, 71, 61, 64, 59, 69, 72, 70, 74, 63, 77, 70, 53, 56, 70, 53, 71, 69, 64, 69, 66, 60, 61, 75, 69, 58, 68, 72, 69, 66, 52, 54, 74, 63, 76, 61, 65, 68, 67, 72, 77, 60, 57, 61, 73, 64, 60, 63, 59, 64, 62, 68, 59, 70, 80, 67, 75, 65, 71, 47, 52, 72, 59, 64, 61, 67, 72, 72, 60, 63, 72, 74, 66, 62, 56, 67, 70, 80, 66, 75, 60, 74, 63, 49, 77, 77, 63, 65, 64, 61, 65, 73, 68, 62, 57, 54, 74, 71, 73, 64, 55, 65, 63, 76, 74, 72, 67, 60, 71, 55, 71, 49, 71, 56, 70, 64, 61, 62, 64, 51, 81, 73, 66, 73, 74, 70],
    'weight': [79.5, 76.75, 95.0, 83.66666666666667, 80.0, 78.0, 61.333333333333336, 95.66666666666667, 66.0, 89.0, 111.0, 78.0, 83.0, 82.6, 111.0, 65.0, 82.0, 77.5, 92.0, 70.0, 91.0, 78.75, 97.0, 82.0, 84.0, 74.0, 80.0, 84.0, 58.0, 78.0, 90.0, 88.0, 85.0, 117.0, 97.0, 72.0, 133.0, 86.0, 108.66666666666667, 100.0, 80.0, 74.5, 75.0, 83.92857142857143, 89.66666666666667, 88.0, 63.0, 48.0, 76.0, 65.4, 70.0, 61.0, 90.2, 92.66666666666667, 80.0, 70.0, 79.75, 95.33333333333333, 54.0, 83.33333333333333, 90.0, 100.0, 92.0, 70.0, 69.0, 78.0, 80.0, 80.0, 105.0, 76.0, 109.0, 85.0, 69.5, 81.5, 57.333333333333336, 92.0, 65.0, 75.66666666666667, 69.0, 74.66666666666667, 81.75, 89.66666666666667, 78.25, 70.0, 65.0, 100.0, 60.0, 66.0, 82.0, 76.0, 108.0, 86.0, 73.0, 65.0, 60.0, 72.0, 83.0, 82.0, 72.66666666666667, 98.0, 103.0, 92.0, 74.0, 101.4, 100.0, 82.0, 80.0, 60.0, 86.0, 74.0, 79.0, 68.0, 87.0, 72.0, 102.0, 95.0, 108.0, 75.0, 115.0, 78.0, 75.0, 109.5, 64.0, 66.0, 90.0, 78.0, 55.585714272090364, 83.0, 77.66666666666667, 105.0, 100.0, 62.0, 76.0, 65.0, 75.0, 80.0, 96.0, 91.0, 73.0, 88.33333333333333, 84.33333333333333, 92.0, 113.8, 67.0, 74.0, 90.0, 95.8, 80.0, 70.0, 78.0, 60.625, 73.0, 67.33333333333333, 73.0, 70.0, 74.0, 74.0, 80.0, 87.5, 153.33333333333334, 78.0, 63.0, 87.33333333333333, 82.0, 68.0, 98.33333333333333, 71.5, 66.33333333333333, 92.0, 80.0, 71.0, 90.0, 78.0, 81.0, 75.5, 69.66666666666667, 78.0, 117.66666666666667, 58.8, 73.0, 100.0, 137.66666666666666, 76.25, 78.25, 76.0, 87.5, 84.0, 95.0, 69.0, 95.0, 89.33333333333333, 76.0, 94.0, 59.333333333333336, 97.0, 66.5, 86.0, 86.0, 81.66666666666667, 83.0, 106.0, 69.33333333333333, 76.0, 74.5, 80.0, 69.66666666666667, 84.0, 78.0, 100.5, 80.0, 84.0, 90.0, 84.66666666666667, 80.0, 60.0, 105.0, 100.0, 135.0, 81.5, 84.33333333333333, 73.5, 85.0, 89.0, 67.0, 69.0, 92.0, 100.0, 82.0, 76.75, 58.0, 78.0, 75.5, 67.5, 86.0, 79.33333333333333, 72.0, 71.5, 75.0, 61.0, 74.0, 65.66666666666667, 62.0, 88.0],
    'height': [176.0, 184.0, 174.0, 171.0, 174.0, 174.0, 176.0, 170.0, 164.0, 176.0, 182.0, 165.0, 178.0, 172.0, 160.0, 160.0, 176.0, 172.0, 183.0, 167.0, 180.0, 172.0, 174.0, 174.0, 180.0, 170.0, 181.5, 168.0, 169.0, 175.0, 188.0, 178.0, 172.0, 182.0, 176.0, 170.0, 175.0, 172.0, 187.0, 175.0, 164.0, 167.0, 168.0, 168.0, 171.0, 178.0, 165.0, 164.0, 176.0, 164.0, 165.0, 173.0, 175.0, 186.0, 170.0, 170.0, 170.0, 174.0, 165.0, 175.0, 171.0, 176.0, 168.0, 165.0, 164.0, 176.0, 178.0, 170.0, 181.0, 174.0, 184.0, 185.0, 165.0, 182.0, 161.0, 176.0, 178.0, 170.0, 169.0, 176.0, 147.59000000357628, 176.0, 168.0, 164.0, 169.0, 180.0, 164.0, 169.0, 176.0, 172.0, 170.0, 178.0, 160.0, 156.0, 170.0, 175.0, 168.0, 170.0, 164.0, 182.0, 172.0, 182.0, 175.0, 172.0, 180.0, 179.0, 174.0, 167.0, 180.0, 176.0, 175.0, 166.0, 174.0, 170.0, 190.0, 172.0, 175.0, 180.0, 174.0, 169.0, 164.0, 180.0, 175.0, 174.0, 173.0, 182.0, 150.9299999986376, 166.0, 167.0, 184.0, 178.0, 170.0, 182.0, 160.0, 176.0, 176.0, 177.0, 165.0, 175.0, 173.0, 190.0, 170.0, 181.0, 169.0, 182.0, 172.0, 170.0, 176.0, 164.0, 170.0, 178.0, 178.0, 179.0, 164.0, 168.0, 176.0, 164.0, 171.0, 170.0, 180.0, 172.0, 164.0, 178.0, 179.0, 168.0, 176.0, 182.0, 164.0, 176.0, 178.0, 174.0, 176.0, 177.0, 172.0, 170.0, 164.0, 180.0, 197.0, 140.9039999961853, 168.0, 178.0, 178.0, 171.0, 174.0, 182.0, 166.0, 180.0, 174.0, 171.0, 176.0, 176.0, 170.0, 176.0, 160.0, 172.0, 171.0, 180.0, 176.0, 180.0, 170.0, 175.0, 178.0, 176.0, 171.0, 172.0, 166.0, 184.0, 178.0, 170.0, 176.0, 182.0, 168.0, 182.0, 170.0, 167.0, 178.0, 188.0, 177.0, 174.5, 170.0, 166.0, 176.0, 174.0, 173.0, 170.0, 186.0, 180.0, 180.0, 179.0, 164.0, 178.0, 172.0, 170.0, 173.0, 165.0, 172.0, 176.0, 162.0, 168.0, 174.0, 171.0, 172.0, 167.0],
    'bmi': [25.66503, 22.66954, 31.37799, 28.61279, 26.42357, 25.76298, 19.80028, 33.10265, 24.53896, 28.73192, 33.51045, 28.65014, 26.19619, 27.9205, 43.35937, 25.39062, 26.47211, 26.19659, 27.47171, 25.0995, 28.08642, 26.61912, 32.03858, 27.08416, 25.92593, 25.60554, 24.28492, 29.7619, 20.30741, 25.46939, 25.46401, 27.77427, 28.73175, 35.32182, 31.31457, 24.91349, 43.42857, 29.06977, 31.07514, 32.65306, 29.7442, 26.71304, 26.57313, 29.7366, 30.66471, 27.77427, 23.1405, 17.84652, 24.53512, 24.31588, 25.71166, 20.38157, 29.45306, 26.78537, 27.68166, 24.22145, 27.59516, 31.48809, 19.83471, 27.21088, 30.7787, 32.28306, 32.59637, 25.71166, 25.65437, 25.18079, 25.24934, 27.68166, 32.0503, 25.10239, 32.19518, 24.83565, 25.52801, 24.60452, 22.11849, 29.70041, 20.51509, 26.18224, 24.15882, 24.10468, 37.5296, 28.94714, 27.72463, 26.02617, 22.75831, 30.8642, 22.30815, 23.10843, 26.47211, 25.68956, 37.37024, 27.14304, 28.51562, 26.7094, 20.76125, 23.5102, 29.4076, 28.3737, 27.01765, 29.5858, 34.81612, 27.77442, 24.16327, 34.27528, 30.8642, 25.59221, 26.42357, 21.51386, 26.54321, 23.88946, 25.79592, 24.67702, 28.73563, 24.91349, 28.25485, 32.11195, 35.26531, 23.14815, 37.98388, 27.30997, 27.88519, 33.7963, 20.89796, 21.79945, 30.07117, 23.54788, 24.40125, 30.12048, 27.84849, 31.01371, 31.56167, 21.45329, 22.94409, 25.39062, 24.21229, 25.82645, 30.64254, 33.42516, 23.83673, 29.51429, 23.36103, 31.83391, 34.73642, 23.45856, 22.3403, 30.42185, 33.14879, 25.82645, 26.02617, 26.98962, 19.13426, 23.04002, 21.01474, 27.14158, 24.80159, 23.88946, 27.51338, 27.35885, 30.27682, 47.3251, 26.3656, 23.42356, 27.56386, 25.59221, 24.09297, 31.74501, 21.58556, 24.6629, 29.70041, 25.24934, 23.45092, 29.05475, 24.89706, 27.37966, 26.12457, 25.90224, 24.07407, 30.31943, 29.61629, 25.86451, 31.56167, 43.4499, 26.0764, 25.84555, 22.94409, 31.75352, 25.92593, 31.37799, 23.597, 30.6689, 28.83953, 26.29758, 30.34607, 23.17708, 32.78799, 22.74204, 26.54321, 27.76343, 25.20576, 28.71972, 34.61224, 21.88276, 24.53512, 25.47792, 27.04164, 25.28185, 24.81096, 24.6181, 34.77509, 25.82645, 25.35926, 31.88776, 25.56052, 27.68166, 21.51386, 33.13976, 28.29335, 43.09107, 26.76497, 29.18108, 26.67296, 27.4406, 29.39622, 22.38631, 23.87543, 26.59267, 30.8642, 25.30864, 23.95368, 21.56454, 24.6181, 25.52055, 23.3564, 28.73467, 29.13988, 24.33748, 23.08239, 28.57796, 21.61281, 24.4418, 22.45705, 20.95727, 31.55366],
    'prostate_volume': [20.25, 26.4, 25.0, 27.0, 44.0, 41.0, 31.0, 41.0, 18.0, 35.0, 37.5, 23.5, 11.5, 15.0, 50.0, 18.0, 17.0, 35.0, 28.0, 33.0, 6.0, 43.333333333333336, 12.0, 51.0, 54.0, 24.0, 32.0, 20.0, 31.0, 37.0, 27.333333333333332, 5.649999976158142, 18.0, 20.5, 39.333333333333336, 85.5, 44.0, 33.0, 36.0, 18.0, 34.0, 18.0, 26.0, 20.0, 14.333333333333334, 15.0, 14.25, 20.0, 41.0, 20.6, 31.0, 31.333333333333332, 17.0, 46.0, 70.0, 36.5, 38.0, 19.166666666666668, 34.0, 29.5, 134.0, 17.300000190734863, 12.0, 39.0, 41.5, 18.0, 32.0, 37.2, 110.5, 38.0, 80.0, 22.0, 29.0, 24.0, 33.333333333333336, 29.0, 27.0, 25.666666666666668, 29.25, 24.0, 34.0, 29.5, 19.0, 17.0, 17.0, 12.0, 46.0, 32.0, 26.0, 24.0, 28.0, 39.333333333333336, 42.0, 78.0, 22.0, 3.0, 1.5, 30.0, 39.5, 17.0, 82.0, 22.0, 25.0, 57.0, 27.0, 42.0, 22.0, 35.0, 28.0, 34.0, 23.0, 38.5, 40.5, 40.0, 24.0, 45.0, 101.0, 23.0, 27.0, 32.0, 22.5, 48.0, 41.0, 96.66666666666667, 26.333333333333332, 36.0, 51.0, 21.333333333333332, 29.25, 45.0, 27.0, 36.0, 29.0, 33.0, 28.0, 36.0, 20.0, 48.0, 19.0, 80.0, 42.333333333333336, 175.0, 32.0, 63.0, 20.0, 26.0, 26.5, 36.0, 14.5, 31.5, 52.0, 25.0, 21.0, 21.5, 39.25, 32.0, 37.0, 24.0, 50.5, 28.0, 62.0, 31.0, 41.0, 30.5, 55.666666666666664, 20.0, 40.0, 29.0, 27.0, 42.0, 23.0, 50.0, 48.0, 39.0, 9.400000095367432, 55.0, 21.5, 66.0, 77.0, 18.0, 26.5, 55.0, 20.0, 27.0, 27.0, 28.0, 28.0, 18.0, 30.0, 20.0, 33.0, 24.0, 30.0, 30.0, 22.0, 38.5, 48.5, 35.0, 28.0, 39.0, 68.0, 39.0, 39.5, 74.0, 8.5, 52.0, 25.0, 30.0, 133.5, 22.0, 80.0, 32.0, 30.0, 31.0, 11.0, 49.0, 37.0, 35.0, 66.5, 65.0, 38.0, 45.5, 30.0, 56.5, 33.0, 36.0, 39.0, 34.5, 26.0, 10.0, 20.0, 70.0, 64.0, 76.5, 37.0, 135.0, 33.0, 45.0, 20.0, 26.0, 55.0, 87.0, 37.0],
    'psa': [2.295, 2.096, 0.37, 1.89, 89.5, 4.56667, 4.2, 0.5, 1.93, 0.6, 2.98, 3.22, 2.855, 450.75, 22.46, 0.18, 0.6, 85.5, 3.155, 12.6, 0.06, 3.09, 0.16, 0.12, 5.655, 0.61, 0.085, 3.55, 23.375, 5.67, 3.94, 1.13, 0.13, 30.0, 7.09333, 113.0, 16.89, 0.38, 1.45, 514.0, 4.79, 32.61, 5875.0, 168.5, 3.85667, 7.72, 1.0375, 3.37, 0.15, 2.018, 5.0, 4.12, 6.67, 54.8, 3.3, 0.495, 3.65, 341.74334, 0.91, 0.63, 0.36, 2.8, 0.5, 5.39, 24.065, 1.31, 0.33, 1.56, 6.06, 27.44, 0.17, 1.75, 10.7, 1.93, 22.08333, 0.26, 0.18, 5.03333, 8.355, 0.23, 5.2, 10.55, 0.35, 0.525, 16.1, 3.59, 68.1, 0.16, 4.56, 10.935, 0.74, 26.05667, 12.9, 0.59, 11.13, 2.28, 3.16, 7.54, 104.1, 5.6, 70.6, 0.09, 13.47, 119.95, 693.0, 1.3, 2.03, 11.235, 2.95, 1.12, 0.34, 5.715, 10.305, 0.25, 7.87, 0.66, 46.0, 1.2, 0.48, 99.9, 0.31, 18.15, 920.0, 9.22667, 0.26333, 0.925, 7.26, 66.64667, 3.205, 2.8, 1.86, 1.495, 29.04, 9.315, 0.52, 12.07, 0.26, 0.35, 0.54, 2099.0, 5.84333, 0.38, 4.32, 0.16, 3.81, 3.02, 2.295, 1.59, 0.255, 6.625, 11.6, 8.81, 1.135, 12.84, 9.565, 0.14, 10.5, 0.8, 5.41, 3.4, 47.305, 0.25, 0.03, 11.825, 3.11667, 0.44, 0.19, 10.57, 4.18, 42.0, 3.87, 0.82, 8.2, 4.355, 0.14, 5.3, 0.27, 22.76, 578.45, 6.0, 3.175, 7.26, 22.1, 3.88, 0.56, 0.15, 10.25, 0.05, 1.04, 49.675, 34.4, 11.46, 188.89999, 0.16, 0.55, 4.65, 14.05, 0.64, 0.07, 3.05, 41.51, 7.12, 0.11, 2.52, 9.54, 3.37, 0.8, 6.0, 81.935, 0.01, 40.71, 1.66, 8.9, 1.97, 11.63, 1.8, 4.66, 2.16, 71.125, 2.63, 1.17, 140.5, 0.23, 62.34, 53.15, 49.1, 2.22, 0.915, 0.28, 0.08, 1.34, 22.12, 9.65, 613.955, 11.42, 14.09, 18.16, 50.54, 4.88, 80.1, 8.56, 9.13, 5.4],
    'density_psa': [0.11333, 0.07939, 0.0148, 0.07, 2.03409, 0.11138, 0.13548, 0.0122, 0.10722, 0.01714, 0.07947, 0.13702, 0.24826, 30.05, 0.4492, 0.01, 0.03529, 2.44286, 0.11268, 0.38182, 0.01, 0.07131, 0.01333, 0.00235, 0.10472, 0.02542, 0.00266, 0.1775, 0.75403, 0.15324, 0.14415, 0.2, 0.00722, 1.46341, 0.18034, 1.32164, 0.38386, 0.01152, 0.04028, 28.55556, 0.14088, 1.81167, 225.96154, 8.425, 0.26907, 0.51467, 0.07281, 0.1685, 0.00366, 0.09796, 0.16129, 0.13149, 0.39235, 1.1913, 0.04714, 0.01356, 0.09605, 17.83009, 0.02676, 0.02136, 0.00269, 0.16185, 0.04167, 0.13821, 0.57988, 0.07278, 0.01031, 0.04194, 0.05484, 0.72211, 0.00213, 0.07955, 0.36897, 0.08042, 0.6625, 0.00897, 0.00667, 0.1961, 0.28564, 0.00958, 0.15294, 0.35763, 0.01842, 0.03088, 0.94706, 0.29917, 1.48043, 0.005, 0.17538, 0.45562, 0.02643, 0.66246, 0.30714, 0.00756, 0.50591, 0.76, 2.10667, 0.25133, 2.63544, 0.32941, 0.86098, 0.00409, 0.5388, 2.10439, 25.66667, 0.03095, 0.09227, 0.321, 0.10536, 0.03294, 0.01478, 0.14844, 0.25444, 0.00625, 0.32792, 0.01467, 0.45545, 0.05217, 0.01778, 3.12188, 0.01378, 0.37812, 22.43902, 0.09545, 0.01, 0.02569, 0.14235, 3.12406, 0.10957, 0.06222, 0.06889, 0.04153, 1.00138, 0.28227, 0.01857, 0.33528, 0.013, 0.00729, 0.02842, 26.2375, 0.13803, 0.00217, 0.135, 0.00254, 0.1905, 0.11615, 0.0866, 0.04417, 0.01759, 0.21032, 0.22308, 0.3524, 0.05405, 0.59721, 0.24369, 0.00438, 0.28378, 0.03333, 0.10713, 0.12143, 0.76298, 0.00806, 0.00073, 0.3877, 0.05599, 0.022, 0.00475, 0.36448, 0.15481, 1.0, 0.16826, 0.0164, 0.17083, 0.11167, 0.01489, 0.09636, 0.01256, 0.34485, 7.51234, 0.33333, 0.11981, 0.132, 1.105, 0.1437, 0.02074, 0.00536, 0.36607, 0.00278, 0.03467, 2.48375, 1.04242, 0.4775, 6.29667, 0.00533, 0.025, 0.12078, 0.28969, 0.01829, 0.0025, 0.07821, 0.61044, 0.18256, 0.00278, 0.03405, 1.12235, 0.06481, 0.032, 0.2, 0.61375, 0.00045, 0.50888, 0.05187, 0.29667, 0.06355, 1.05727, 0.03673, 0.12595, 0.06171, 1.06955, 0.04046, 0.03079, 3.08791, 0.00767, 1.10336, 1.61061, 1.36389, 0.05692, 0.02652, 0.01077, 0.008, 0.067, 0.316, 0.15078, 8.02556, 0.30865, 0.10437, 0.5503, 1.12311, 0.244, 3.08077, 0.15564, 0.10494, 0.14595],
    'T': ['3a', '2', '3b', '3', '3b', '3', '3b', '2', '3b', '3a', '3b', '3b', '3b', '3', '3b', '3', '2', '3', '3', '3b', '3a', '3b', '2', '2', '2', '2', '2', '3', '3', '2b', '3', '2c', '2', '3b', '3', '3', '2', '3b', '2b', '3', '3', '3', '2c', '3b', '4', '3b', '2c', '3b', '4', '2', '2', '3', '4', '2', '2c', '3', '2c', '2c', '3', '3b', '2', '2', '2', '2b', '3a', '2', '1', '3a', '2', '3b', '3', '4', '3b', '2c', '2b', '4', '3', '3b', '4', '3a', '2', '2', '3', '2', '2c', '3', '3', '3', '2', '3', '3', '3', '3', '2', '2', '2', '3', '3', '3', '2', '3', '2', '2', '3', '3', '4', '4', '1', '3', '3', '3', '3', '2', '3', '2', '3', '3', '3', '2', '2', '2', '3', '3', '2', '3', '3', '2', '4', '3', '3', '2', '3', '4', '2', '3', '3', '3', '3', '4', '3', '2', '2', '2', '3', '2', '3', '3', '3', '2', '2', '2', '2', '3', '3', '3', '2', '3', '4', '3', '3', '3', '2', '3', '3', '2', '2', '3', '3', '2', '4', '3', '2', '2', '3', '2', '2', '2', '3', '3', '2', '3', '2', '4', '3', '3', '3', '1', '3', '2', '2', '2', '2', '2', '2', '3', '2', '2', '3', '2', '2', '3', '2', '2', '2', '3', '3', '2', '3', '3', '3', '2', '4', '1', '3', '2', '2', '3', '3', '2', '3', '2', '2', '2', '4', '3', '3', '3', '2', '4', '3', '3', '3', '1', '4', '3', '3', '3', '2', '3', '2', '2', '2', '3'],
    'N': ['0', '0', '1', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '1', '0', '1', '0', '0', '0', '0', '1', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '1', '0', '0', '0', '1', '1', '0', '0', '0', '1', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '1', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '1', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
    'M': ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1b', '1b', '0', '0', '1b', '0', '1b', '0', '0', '0', '0', '0', '0', '1a', '1b', '0', '1b', '0', '0', '0', '1b', '1b', '1b', '0', '0', '0', '1b', '1b', 'a+b', '1b', 'a+b+c', '1a', '0', '0', '0', '0', '0', '0', '0', '1b', '1b', '0', '0', '0', 'a+b', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1b', '0', '1b', '0', '0', '0', '0', '1b', '0', '1b', '0', '0', '0', '1b', '0', '1b', '0', '0', '1b', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1b', '1b', '0', '1b', 'b+c', '0', '1b', '0', '0', '1b', '0', '1a', '0', '0', '0', '1b', '1b', '0', '0', '1b', '0', '0', '1b', '0', '0', '0', '0', '1b', '0', '0', '1b', '0', '0', '0', '0', '1b', '0', '0', '0', 'a+b+c', '0', '0', '1b', '0', '0', '0', '0', '0', '0', '1b', '0', '0', '1b', '0', '0', '1b', '0', '0', '1b', '1b', '0', '0', '0', '0', '0', '0', '1b', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1b', '0', '1b', '1b', '0', '0', '1b', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1b', '0', '0', '0', '0', '1b', '0', '0', '0', '0', '0', '0', '0', '1b', '0', '0', '1b', '0', '0', '0', '0', '0', '0', '0', '0', '1b', '0', '0', '0', '0', '0', '1b', '1b', '1b', '0', '0', '0', '0', '0', '0', '0', 'a+b+c', '1b', '0', '0', '0', '0', '0', '0', '0', '0'],
    'G': [6.0, 6.0, 8.0, 7.0, 7.0, 6.0, 6.0, 6.0, 9.0, 6.0, 6.0, 9.0, 6.0, 7.0, 7.0, 8.0, 4.0, 9.0, 9.0, 9.0, 7.0, 7.0, 7.0, 8.0, 7.0, 9.0, 7.0, 6.0, 7.0, 8.0, 9.0, 7.0, 7.0, 8.0, 6.0, 4.0, 6.0, 7.0, 6.0, 6.0, 7.0, 8.0, 7.0, 8.0, 6.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 5.0, 9.0, 8.0, 7.0, 6.0, 8.0, 8.0, 8.0, 8.0, 7.0, 8.0, 7.0, 8.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 5.0, 6.0, 8.0, 7.0, 7.0, 9.0, 8.0, 8.0, 5.0, 5.0, 7.0, 7.0, 8.0, 9.0, 7.0, 9.0, 7.0, 9.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 8.0, 8.0, 6.0, 8.0, 7.0, 9.0, 8.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 6.0, 7.0, 8.0, 6.0, 8.0, 7.0, 7.0, 6.0, 8.0, 9.0, 7.0, 7.0, 10.0, 6.0, 8.0, 6.0, 8.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0, 8.0, 7.0, 7.0, 5.0, 7.0, 8.0, 7.0, 7.0, 8.0, 6.0, 7.0, 7.0, 7.0, 7.0, 6.0, 6.0, 7.0, 6.0, 8.0, 6.0, 7.0, 8.0, 9.0, 10.0, 7.0, 6.0, 7.0, 6.0, 6.0, 8.0, 8.0, 8.0, 7.0, 6.0, 8.0, 8.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 8.0, 6.0, 9.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 6.0, 9.0, 4.0, 7.0, 7.0, 7.0, 8.0, 7.0, 7.0, 8.0, 6.0, 7.0, 6.0, 6.0, 7.0, 7.0, 7.0, 6.0, 8.0, 7.0, 7.0, 8.0, 8.0, 7.0, 8.0, 8.0, 8.0, 8.0, 7.0, 8.0, 6.0, 8.0, 8.0, 6.0, 10.0, 7.0, 7.0, 8.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0]
})

df_skill.to_sql('PATIENT', con)

con.close()