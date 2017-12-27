# Author: Nishank Bhatnagar
# Machine Learning: Implements K_Means clustering algorithm, uses Elbow Test to find the right K value
# Evaluates the algorithm on Iris Dataset

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self, data_x_point,data_y_point):
        self.data_x_point = data_x_point
        self.data_y_point = data_y_point

    """
        This function is used to show the plotting of graphs for elbow test and k means 
    """
    def plot_centers(self,center_info, pred_table):

        color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'cyan', 5: 'magenta', 6: 'yellow', 7: 'brown',
                     8: 'pink', 9: 'grey'}

        for kval, cent_val in center_info.items():
            x_points = pred_table[pred_table['closest_to'] == kval]['x_val']
            y_points = pred_table[pred_table['closest_to'] == kval]['y_val']
            plt.scatter(x_points, y_points, c=color_map[kval])
            plt.scatter(cent_val[0], cent_val[1], s=100, marker="*", c="black")

        plt.show()

    """
        Initialises the values of K Centroids from randomly selecting K values from 
        the given data points  
    """
    def initailize_k(self,data_val_x, data_val_y, k_value):
        print("Centroid initialized ....... ")
        rand_initializer = [(i, np.random.randint(1, 150)) for i in range(k_value)]
        centroid = {
            r_val[0]: [data_val_x[r_val[1]], data_val_y[r_val[1]]]
            for r_val in rand_initializer
        }

        print(centroid)
        print("================================================")
        return centroid

    """
        Calculate the Euclidean distance between the input points and k centroid values 
    """
    def euclidean_distance(self,data_x_value, data_y_value, centroids):
        # Create a new pandas empty Data Frame table
        tbl = pd.DataFrame()
        closest_val = []

        # Calculating the Euclidean distance of each input X and Y with centroid Ck
        for k, val in centroids.items():
            tbl["distance from centroid {}".format(k)] = (
                np.sqrt(
                    (data_x_value - val[0]) ** 2
                    + (data_y_value - val[1]) ** 2
                )
            )

        # Get the Column values from the table
        cols_values = list(tbl.columns.values)

        # Selecting all the rows with calculated Euclidean distance and getting the column value
        # with minimum distance value
        tbl['closest_to'] = tbl.loc[:, cols_values].idxmin(axis=1)
        tbl['closest_to'] = tbl['closest_to'].apply(lambda cls: int(cls.lstrip('distance from centroid ')))

        # Adding X value and Y value dataset to the table before returning it
        tbl['x_val'] = data_x_value
        tbl['y_val'] = data_y_value

        return tbl

    """
        Updates the centroid by calculating the mean value of corresponding inputs from each clusters
    """
    def update_centroid(self,dist_table, cent):
        new_centroids = cent
        for kval, centers in new_centroids.items():
            # Updates the Centroid by getting the mean value for X val
            new_centroids[kval][0] = np.mean(dist_table[dist_table['closest_to'] == kval]['x_val'])
            # Updates the Centroid by getting the mean value for Y val
            new_centroids[kval][1] = np.mean(dist_table[dist_table['closest_to'] == kval]['y_val'])

        return new_centroids

    """
        calculates the sum of squared error for Elbow test 
    """
    def sum_of_squared_error(self,center_info, result_table):
        f_k_vals = []
        for kval, value in center_info.items():
            x_points = result_table[result_table['closest_to'] == kval]['x_val']
            y_points = result_table[result_table['closest_to'] == kval]['y_val']
            f_k_vals.append(np.sum(np.power((x_points - value[0]), 2) + np.power((y_points - value[1]), 2)))
        f_val = np.sum(f_k_vals)
        return f_val

    """
        This is the main function which takes in K value and performs the K means algorithm
    """
    def K_means(self, k_cluster=3, elbow_test=False):
        print("K Means Clustering Executed......")
        print("================================================")
        centroids_info = self.initailize_k(self.data_x_point, self.data_y_point, k_cluster)

        while True:
            flag = True
            # need to take deep copy else old centroid gets updated along as well and comparison cannot be done properly
            old_cent = copy.deepcopy(centroids_info)
            result_table = self.euclidean_distance(self.data_x_point, self.data_y_point, centroids_info)
            centroids_info = self.update_centroid(result_table, centroids_info)

            if not elbow_test:
                print("Old Centroid")
                print(old_cent)
                print("New Centroid")
                print(centroids_info)
                self.plot_centers(old_cent, result_table)
                print("======================================================================================")

            # Checks if the centroid is still being changed as long as there is still a change flag will remain false
            for k in centroids_info.keys():
                if (old_cent[k] == centroids_info[k]) == False:
                    flag = False

            if flag:
                break

        print("******************** End of Kmeans ***********************")

        sse_val = self.sum_of_squared_error(centroids_info, result_table)

        return (centroids_info, result_table, sse_val)



if __name__ == '__main__':
    col_name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    kvalue = 3

    # Replaces the Class Value from String to Numerical Value
    class_val = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    data_iris = pd.read_csv("iris_dataset.csv", names=col_name, header=None)
    data_iris = data_iris.replace({'class': class_val})
    print("The DataSet Iris :- ")
    print(data_iris)

    print("IrisDataSet Loaded ......")
    print("================================================")
    sep_length = data_iris['sepal-length'].values
    sep_width = data_iris['sepal-width'].values
    pet_length = data_iris['petal-length'].values
    pet_width = data_iris['petal-width'].values

    # Create the KMeans Class object
    kmean = KMeans(pet_length, pet_width)

    print("Performing Elbow Method Test ......")
    print("================================================\n\n")
    k_vals = [1,2,3,5,6,7,9]
    sse_error = []

    for k in k_vals:
        info = kmean.K_means(k, elbow_test=True)
        print(info[0])
        sse_error.append(info[2])

    print("\n\n")
    print("================================================================================================")
    print(" List of Sum of squared error " ,sse_error)
    print("================================================================================================\n\n")

    plt.title("Elbow Test", fontsize=20)
    plt.xlabel("K Value", fontsize=16)
    plt.ylabel("SSE Value (sum of squared)", fontsize=16)
    plt.plot(k_vals, sse_error, marker='s')
    plt.legend(["Sum of squared value in terms of k"])
    plt.show()

    # Based on the result of Elbow test
    center_info, predicted_table, sse_val = kmean.K_means(k_cluster=kvalue)
    print(predicted_table)

    # Puts the predicted file into a csv file
    predicted_table.to_csv("resultant_file.csv")
    print("================================================================================================")
    print(" File resultant_file.csv created")
    print("================================================================================================\n\n")

    if kvalue == 3:
        predicted = np.sort(predicted_table['closest_to'])
        true = np.sort(data_iris['class'])
        acc_vals = predicted - true
        cnt_nonzeroes = np.count_nonzero(acc_vals)
        cnt_zeroes = len(acc_vals) - cnt_nonzeroes
        accuracy = (cnt_zeroes / len(acc_vals)) * 100
        print("================================================================================================")
        print("The accuracy achieved with clustering is {}%".format(accuracy))
        print("================================================================================================\n\n")
