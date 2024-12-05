import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud("merged_offices.ply")  # Replace with your file path
print("Loaded point cloud:", pcd)

# Visualize the original point cloud
o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

planes = []  # List to store the detected planes
plane_models = []  # List to store plane equations
remaining_cloud = pcd

# Parameters for RANSAC
distance_threshold = 0.01  # Adjust based on your data
ransac_n = 3
num_iterations = 1000

# Detect 3 main planes iteratively
for i in range(3):
    print(f"Detecting plane {i + 1}")
    
    # Segment the largest plane in the current point cloud
    plane_model, inliers = remaining_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    plane_models.append(plane_model)
    print(f"Plane {i + 1} equation: {plane_model}")
    
    # Extract inlier and outlier point clouds
    inlier_cloud = remaining_cloud.select_by_index(inliers)
    outlier_cloud = remaining_cloud.select_by_index(inliers, invert=True)
    
    # Color the detected plane for visualization
    inlier_cloud.paint_uniform_color(np.random.rand(3))
    planes.append(inlier_cloud)
    
    # Update the remaining cloud for the next iteration
    remaining_cloud = outlier_cloud

    # Visualize the segmented plane and remaining cloud
    o3d.visualization.draw_geometries(
        [inlier_cloud, outlier_cloud],
        window_name=f"Plane {i + 1} and Remaining Cloud"
    )

# Visualize all detected planes
o3d.visualization.draw_geometries(
    planes, window_name="Detected Planes"
)

print("Plane segmentation completed.")
