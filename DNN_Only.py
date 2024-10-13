import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, NearestNDInterpolator
import time

sdf_dir = os.path.abspath(os.path.join('..', 'DNN/UI'))
sys.path.append(sdf_dir)

dnn_dir = os.path.abspath(os.path.join('..', 'DNN/UI'))
sys.path.append(dnn_dir)

from sdf_generator import generate_sdf

from DNN_UI import ResidualBlock, ChannelSpecificDecoder, EncoderDecoderCNN, model_init, main_inference # Need to call first three in main(console) everytime

def plot_pressure_distribution_and_boundary(p_norm, boundary_points, x_smooth, y_smooth, p_surface, x_range, y_range, airfoil_coords, grid_x, grid_y, grid_size=150 ):
    
    # Plot the normalized pressure field
    plt.figure(figsize=(12, 6))
    plt.contourf(grid_x, grid_y, p_norm, levels=200, cmap='turbo')
    plt.colorbar(label='Normalized Pressure')
    plt.title('Pressure Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.show()
    plt.show()
    
    # Plot the detected boundary points
    #plt.figure(figsize=(12, 6))
    #plt.scatter(boundary_points[:, 1], boundary_points[:, 0], color='red', s=5)
    #plt.title('Boundary Points')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.grid(False)
    #plt.show()
    
    # Plot the smoothed boundary curve
    #plt.figure(figsize=(12, 6))
    #plt.plot(x_smooth, y_smooth)
    #plt.title('Smoothed Boundary')
    #plt.show()
    
    # Plot pressure on boundary points
    plt.figure(figsize=(12, 6))
    p_boundary_norm = p_norm[boundary_points[:, 0], boundary_points[:, 1]]
    plt.scatter(boundary_points[:, 1].astype(int), boundary_points[:, 0].astype(int), c=p_boundary_norm, cmap='turbo')
    plt.colorbar(label='Normalized Pressure')
    plt.title('Pressure at Boundary Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.show()
    
    # Plot pressure on smoothed boundary
    plt.figure(figsize=(12, 6))
    plt.scatter(x_smooth, y_smooth, c=p_surface, cmap='turbo')
    plt.colorbar(label='Normalized Pressure')
    plt.title('Pressure on Smoothed Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(False)
    plt.show()
    
    # Conversion factors from grid to original coordinates
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Convert grid indices back to original coordinates
    boundary_points_physical_x = x_min + (boundary_points[:, 1] / (grid_size - 1)) * (x_max - x_min)
    boundary_points_physical_y = y_min + (boundary_points[:, 0] / (grid_size - 1)) * (y_max - y_min)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_smooth, y_smooth, label='Airfoil Geometry')
    plt.scatter(boundary_points_physical_x, boundary_points_physical_y, color='red', s=2, label='Boundary Points')
    #plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], 'k', lw=2, label='Airfoil Geometry')
    plt.title('Airfoil Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(False)
    plt.show()
    
#.....................................................................................................................................................
#1.SDF Generator

# SDF generator module imported

#.................................................................................................................................................
#1.5. DNN Inference

# DNN module imported

#.................................................................................................................................................
# 2. Process Inferred Field
def load_inferred_field(airfoil_number, mach, aoa, inferred_data_folder):
    inferred_file_path = os.path.join(inferred_data_folder, f'{airfoil_number}_{mach}_{aoa}.npy')
    inferred_field = np.load(inferred_file_path)
    return inferred_field

def introduce_airfoil_geometry(inferred_field, sdf):
    # Set values to NaN for points inside the airfoil
    inferred_field[sdf < 0] = np.nan
    return inferred_field

def load_global_min_max():
    global_min_max_folder = './'
    global_min_path = os.path.join(global_min_max_folder, 'global_min.npy')
    global_max_path = os.path.join(global_min_max_folder, 'global_max.npy')
    
    global_min = np.load(global_min_path)
    global_max = np.load(global_max_path)
    
    return global_min, global_max

def denormalize_inferred_field(inferred_field, global_min, global_max):
    denormalized_field = np.empty_like(inferred_field)
    
    for i in range(inferred_field.shape[2]):
        denormalized_field[:, :, i] = inferred_field[:, :, i] * (global_max[i] - global_min[i]) + global_min[i]
    
    denormalized_field[np.isnan(inferred_field)] = np.nan
    return denormalized_field

def plot_denormalized_field(denormalized_field, airfoil_number, grid_x, grid_y):
    channels = denormalized_field.shape[2]
    fig, axes = plt.subplots(1, channels, figsize=(20, 5))
    for i in range(channels):
        im = axes[i].contourf(grid_x, grid_y, denormalized_field[:, :, i], levels=200, cmap='turbo')
        fig.colorbar(im, ax=axes[i])
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].grid(False)
    plt.suptitle(f'Denormalized Field for Airfoil {airfoil_number}')
    plt.tight_layout()
    plt.show()

def process_inferred_field(sdf_image, airfoil_number, mach, aoa, output_dir):
    start_m = time.time()
    
    inferred_field = load_inferred_field(airfoil_number, mach, aoa, output_dir)
    inferred_field = introduce_airfoil_geometry(inferred_field, sdf_image)
    global_min, global_max = load_global_min_max()
    denormalized_field = denormalize_inferred_field(inferred_field, global_min, global_max)
    
    end_m = time.time()
    elapsed_total = end_m - start_m
    print(f"Inferred Field Processing Time: {elapsed_total:.2f} seconds")
    return denormalized_field

#.................................................................................................................................................
# 3. Boundary detection
def find_boundary_from_field(denormalized_field):
    valid_mask = ~np.isnan(denormalized_field)
    boundary_mask = np.zeros_like(valid_mask[:, :, 0], dtype=bool)
    
    for i in range(1, valid_mask.shape[0] - 1):
        for j in range(1, valid_mask.shape[1] - 1):
            if valid_mask[i, j, 0]:
                neighbors = valid_mask[i-1:i+2, j-1:j+2, 0]
                if np.any(~neighbors):
                    boundary_mask[i, j] = True
    
    boundary_points = np.column_stack(np.where(boundary_mask))
    
    return boundary_points, boundary_mask

def plot_boundary_detection(denormalized_field, boundary_points, airfoil_coords):
    plt.figure(figsize=(12, 6))
    
    plt.imshow(denormalized_field[:, :, 0], cmap='turbo', origin='lower')
    plt.colorbar(label='Denormalized Field Value')
    plt.scatter(boundary_points[:, 1], boundary_points[:, 0], color='red', s=1, label='Detected Boundary')
    plt.title('Detected Boundary')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(False)
    plt.gca().invert_yaxis()  
    plt.ylim(0, denormalized_field.shape[0])  
    plt.show()
    
#.................................................................................................................................................
# 4. Pressure calculation
def calculate_normalized_pressure(flow_field, gamma=1.4):
    rho_star = flow_field[:, :, 0]
    u = flow_field[:, :, 1]
    v = flow_field[:, :, 2]
    internal_energy = flow_field[:, :, 3]

    T_star = (internal_energy - (u**2 + v**2)) * gamma
    p_star = ((gamma - 1)/ gamma) * rho_star * T_star
    p_norm = 2.0 * p_star
    
    return p_norm

def interpolate_pressure_along_surface(x_smooth, y_smooth, boundary_points, p_norm, x_range, y_range, grid_size=150):
    
    # Extract pressure values at boundary points
    p_boundary_norm = p_norm[boundary_points[:, 0].astype(int), boundary_points[:, 1].astype(int)]
    
    # Conversion factors from grid to original coordinates
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Convert grid indices back to original coordinates
    x_boundary = x_min + (boundary_points[:, 1] / (grid_size - 1)) * (x_max - x_min)
    y_boundary = y_min + (boundary_points[:, 0] / (grid_size - 1)) * (y_max - y_min)
    
    # Perform interpolation at the smooth curve points
    p_surface = griddata(
        (x_boundary, y_boundary),  # Points to interpolate from
        p_boundary_norm,  # Values to interpolate
        (x_smooth, y_smooth),  # Points to interpolate to
        method='nearest'  # Interpolation method
    )
    
    # Identify NaN values in the interpolated pressure surface
    nan_indices = np.isnan(p_surface)
    
    # Use nearest neighbor interpolation to fill NaN values
    if np.any(nan_indices):
        # Create a nearest neighbor interpolator
        nearest_interp = NearestNDInterpolator(
            (x_boundary, y_boundary), p_boundary_norm
        )
        
        # Fill NaN values with the nearest neighbor values
        p_surface[nan_indices] = nearest_interp(x_smooth[nan_indices], y_smooth[nan_indices])
        
    return p_surface

#.................................................................................................................................................
# 5. Coefficients calculation
def calculate_forces_and_coefficients(denormalized_field, alpha, airfoil_coords, x_range, y_range, grid_size=150, gamma=1.4):
    start_m = time.time()
    
    # Step 1: Normalize pressure
    p_norm = calculate_normalized_pressure(denormalized_field, gamma)
    
    # Step 2: Find boundary points
    boundary_points, _ = find_boundary_from_field(denormalized_field)
    
    if boundary_points.size == 0:
        raise ValueError("No boundary points found.")
    
    # Use original airfoil coordinates to create a smooth curve
    x_smooth, y_smooth = airfoil_coords[:, 0], airfoil_coords[:, 1]
    
    # Step 3: Interpolate pressure along the smooth surface
    p_surface = interpolate_pressure_along_surface(x_smooth, y_smooth, boundary_points, p_norm, x_range, y_range)
#.....................................................................................................................................................    
    # Step 4: Calculate forces using pressure on the smooth surface
    CY = 0  # Lift
    CX = 0  # Drag

    for i in range(1, len(x_smooth)):
        dx = x_smooth[i] - x_smooth[i-1]
        dy = y_smooth[i] - y_smooth[i-1]

        # Segment length
        ds = np.sqrt(dx**2 + dy**2)
        if ds == 0:
            continue  # Skip if segment length is zero to avoid division by zero

        # Normal vector (perpendicular to the surface, inward pointing)
        nx = dy / ds
        ny = -dx / ds

        # Pressure difference across the segment
        p_avg = (p_surface[i] + p_surface[i-1]) / 2

        # Force contributions
        F_x = -p_avg * ds * nx
        F_y = -p_avg * ds * ny

        # Ensure that force contributions are not NaN before adding
        if not np.isnan(F_x) and not np.isnan(F_y):
            CX += F_x
            CY += F_y

    # Step 5: Output coefficients
    #print(f"Force_X Coefficient: {CX}")
    #print(f"Force_Y Coefficient: {CY}")
    
    # Convert angle of attack from degrees to radians
    #alpha_rad = np.radians(alpha)

    # Calculate lift coefficient (CL) and drag coefficient (CD)
    #C_L = CY * np.cos(alpha_rad) - CX * np.sin(alpha_rad)
    #C_D = CY * np.sin(alpha_rad) + CX * np.cos(alpha_rad)
    C_L = CY
    C_D = -CX
    
    end_m = time.time()
    elapsed_total = end_m - start_m
    print(f"Time taken to calculate coefficients: {elapsed_total:.2f} seconds")
    
    return C_L, C_D, p_norm, x_smooth, y_smooth, p_surface, boundary_points

def workflow(airfoil_number, mach, aoa, sdf_path = './sdf', geo_path = './data_geometry', output_dir = './inferred_data'):
    start_total = time.time()
    
    # Generate SDF
    airfoil_coords, sdf_image, grid_x, grid_y, x_range, y_range = generate_sdf(airfoil_number, sdf_path, geo_path)

    model = model_init()

    main_inference(airfoil_number, mach, aoa, model, sdf_path, output_dir)

    # Process inferred field
    denormalized_field = process_inferred_field(sdf_image, airfoil_number, mach, aoa, output_dir)
    #plot_denormalized_field(denormalized_field, airfoil_number, grid_x, grid_y)

    # Calculate forces and coefficients
    C_L, C_D, p_norm, x_smooth_physical, y_smooth_physical, p_surface, boundary_points = calculate_forces_and_coefficients(denormalized_field, aoa, airfoil_coords, x_range, y_range)

    print(f"Lift Coefficient (C_L): {C_L}")
    print(f"Drag Coefficient (C_D): {C_D}")

    # Plot the normalized pressure field, boundary points, smooth boundary, and pressures
    #plot_boundary_detection(denormalized_field, boundary_points, airfoil_coords)
    #plot_pressure_distribution_and_boundary(p_norm, boundary_points, x_smooth_physical, y_smooth_physical, p_surface, x_range, y_range, airfoil_coords, grid_x, grid_y)
    
    # Total execution time
    end_total = time.time()
    elapsed_total = end_total - start_total
    print(f'Total time taken: {elapsed_total:.2f} seconds')
    
    return C_L, C_D
#Example Implementation in console
#from DNN_Only import ResidualBlock, ChannelSpecificDecoder, EncoderDecoderCNN, workflow
#workflow(58,0.6,2)