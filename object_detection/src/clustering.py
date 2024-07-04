#!/usr/bin/env python
"""
Read the csv file and cluster the co-ordinate points
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tkinter as tk
from tkinter import simpledialog

OMP_NUM_THREADS=1

class Clustering:
    def __init__(self, csv_file='object_detection.csv'):
        self.data = pd.read_csv(csv_file)
        self.X = self.data[['class', 'conf', 'x', 'y', 'z']]
        self.cluster_centers = {}
        self.detections = {}
        self.num_clusters_each_class = {}
        
        for i in range(len(self.data)):
            class_name = self.data['class'][i]
            conf = self.data['conf'][i]
            x = self.data['x'][i]
            y = self.data['y'][i]
            z = self.data['z'][i]
    
            if class_name not in self.detections:
                self.detections[class_name] = []
            self.detections[class_name].append([x, y, z, conf])
        
        self.classes = list(self.detections.keys())
        print(f"Classes detected: {self.classes}")

        # Print in dictionary format so that it can be used later
        dict_str = '{'
        for class_name in self.classes:
            dict_str += f"'{class_name}': [],\n"
        dict_str += '}'
        print(dict_str)

        # self.X = StandardScaler().fit_transform(self.X)
        
    def plot(self):
        """Plot the 2D data points as a scatter plot.
        Each class is represented by a different color.
        Transparency is used to show the confidence level of the detection.
        """

        # Plot the 2D data points
        plt.figure()
        for class_name, det in self.detections.items():
            det = np.array(det)
            x = det[:, 0]
            y = det[:, 1]
            conf = det[:, 2]

            plt.scatter(x, y, alpha=conf, label=class_name)

        # Plot the cluster centers as stars
        if hasattr(self, 'cluster_centers'):
            for class_name, centers in self.cluster_centers.items():
                plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, label=f'{class_name} center')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def get_unique_class_name(self,class_name):
        original_name = class_name
        count = 1
        while class_name in self.cluster_centers:
            class_name = f"{original_name}_{count}"
            count += 1
        # how many cluster centers belong to the same class
        self.num_clusters_each_class[original_name] = count 
        return class_name
    
    def filter_outliers(self,data_points, cluster_centers, threshold):
        filtered_points = []
        for point in data_points:
            min_distance = min(np.linalg.norm(np.array(point[:2]) - np.array(center)) for center in cluster_centers)
            if min_distance <= threshold:
                filtered_points.append(point)
        return np.array(filtered_points)
    
    
    def interactive_plot(self):
        """
        Take use prompt from the input, and save the input class, clicked location in a csv file.
        """
        def on_click(event):
            
            if event.inaxes:
                x, y = event.xdata, event.ydata
                root = tk.Tk()
                root.withdraw()
                class_name = simpledialog.askstring("Input", f"Enter class name for the point ({x:.2f}, {y:.2f}):")
                
                # Only save if the class name is provided
                if class_name:
                    # append a number if class name already exists
                    unique_class_name = self.get_unique_class_name(class_name)
                    self.cluster_centers[unique_class_name] = [x,y]
                    print(f"Saved: ({x:.2f}, {y:.2f}) -> {class_name}")
                root.destroy()
        
        fig, ax = plt.subplots()
        for class_name, det in self.detections.items():
            det = np.array(det)
            x = det[:, 0]
            y = det[:, 1]
            conf = det[:, 3]

            ax.scatter(x, y, alpha=conf, label=class_name)

        # Plot the cluster centers as stars
        if hasattr(self, 'cluster_centers'):
            for class_name, centers in self.cluster_centers.items():
                ax.scatter(centers[0], centers[1], marker='*', s=200, label=f'{class_name} center')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()

        # click and prompt function
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        fig.canvas.mpl_disconnect(cid)  
        
        # Plot the cluster centers as stars
        if hasattr(self, 'cluster_centers'):
            for class_name, centers in self.cluster_centers.items():
                ax.scatter(centers[0], centers[1], marker='*', s=200, label=f'{class_name} center',color='red')
                
        plt.show()
    
    
    def cluster_new(self):
        centers = self.cluster_centers
        detections = self.detections
        
        final_cluster_centers = {}
        for class_name, points in detections.items():
            center_list = []
            # xy_points = np.array(points)[:, :2]
            # print(xy_points)
            for center in centers.keys():
                if class_name in center:
                    center_list.append(centers[class_name])
                    
            num_clusters_in_class = self.num_clusters_each_class.get(class_name, 0)
            if num_clusters_in_class == 0:
                continue
            center_arr = np.array(center_list)
            filtered_points = self.filter_outliers(points,center_list,2.5)
            filtered_xy_points = filtered_points[:,:2]
            kmeans = KMeans(n_clusters=num_clusters_in_class, init=center_arr, n_init=1)
            kmeans.fit(filtered_xy_points)

            # Get cluster labels and centroids
            cluster_labels = kmeans.labels_
            cluster_centers = kmeans.cluster_centers_
            
            # Add z coordinates to data points with cluster labels
            filtered_xyz_points = filtered_points[:,:3]
            data_with_clusters = np.column_stack((filtered_xyz_points, cluster_labels))

            # Initialize array to store mean z values for each cluster
            mean_z_values = np.zeros(num_clusters_in_class)
            
            # Compute mean z for each cluster
            for cluster in range(num_clusters_in_class):
                cluster_data = data_with_clusters[data_with_clusters[:, -1] == cluster]
                mean_z_values[cluster] = np.mean(cluster_data[:, 2])  # Mean of z values in this cluster

            cluster_centers_with_z = np.column_stack((cluster_centers, mean_z_values))
            
            # print(cluster_centers_with_z)
            
            for i in range(cluster_centers_with_z.shape[0]):
                # append a number if class name already exists
                unique_class_name = class_name + f"_{i}"
                final_cluster_centers[unique_class_name] = cluster_centers_with_z[i,:]
            
            
        return final_cluster_centers
            
    # NOT USED
    def cluster(self,maxClusters=10):
        """
        Cluster the data points using KMeans clustering.
        Number of clusters is determined using silhouette score if centers is None.
        """
        
        # Convert centers to array to pass as initialization for kmeans
        # centers_list = []
        # for class_name, coordinates in self.cluster_centers.items():
        #     centers_list.append(coordinates)
        # centers_arr = np.array(centers_list)
            
        for class_name in self.classes:
            if class_name not in centers:
                # Find the optimal number of clusters
                silhouette_scores = []
                coords = np.array(self.detections[class_name])[:, :2]

                if len(coords) < 2:
                    centers[class_name] = 1
                    continue

                for n_clusters in range(2, min(len(coords), maxClusters)):
                    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
                    kmeans.fit(coords)
                    labels = kmeans.labels_
                    silhouette_scores.append(silhouette_score(coords, labels))
                
                if len(silhouette_scores) == 0:
                    centers[class_name] = 2
                    continue

                n_clusters = np.argmax(silhouette_scores) + 2

                centers[class_name] = n_clusters
        
        cluster_centers = {}
        for class_name, n_clusters in centers.items():
            coords = np.array(self.detections[class_name])[:, :2]
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
            kmeans.fit(coords)
            cluster_centers[class_name] = kmeans.cluster_centers_
            

        # self.cluster_centers = cluster_centers

    def save_centers(self, final_cluster_centers):
        # save the clustered centers in a new .csv file
        header = ['class', 'x', 'y', 'z']
        data = []
        for class_name, center in final_cluster_centers.items():
            data.append([class_name, center[0], center[1], center[2]])
        df = pd.DataFrame(data, columns=header)
        df.to_csv('cluster_centers.csv', index=False)

if __name__ == '__main__':
    csv_file = 'object_detection.csv'
    clustering = Clustering(csv_file)
    # clustering.plot()
    clustering.interactive_plot()
    clustering.interactive_plot()

    # centers = {'stop sign': 1,
    #             'backpack': 1,
    #             'umbrella': 1,
    #             'clock': 1,
    #             }
    
    final_cluster_centers = clustering.cluster_new()
    # clustering.plot()

    clustering.save_centers(final_cluster_centers)


