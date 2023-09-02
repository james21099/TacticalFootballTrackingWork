"""
This file creates all the fucntions which will be used for
analysis of this game, it will analyse, the shape teams are in
within each phase, the lengths of passes, the lengths of phases
in no. of passes and in time, the intensty of teams running,
and the extent of their man oreinted pressing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import ConvexHull
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import euclidean
import sys
sys.path.append('/Users/jamesgothard/Research Project/Metrica/Python files')
plt.rcParams["font.family"] = "Times New Roman"


# Create the 11 formation templates
# Each player is a tuple of coordinates (x, y)
formations = {
    "3-2-5": [
        # Three defenders
        (-30, -10), (-30, 0), (-30, 10),
        # Two midfielders
        (-20, -5), (-20, 5),
        # Five forwards
        (-10, -23), (-10, -10), (-10, 0), (-10, 10), (-10, 23),
    ],
    "3-4-3": [
        # Three defenders
        (-30, -10), (-30, 0), (-30, 10),
        # Four midfielders
        (-20, -23), (-20, -5), (-20, 5), (-20, 23),
        # Three forwards
        (-12, -10), (-10, 0), (-12, 10),
    ],
    "3-5-2": [
        # Three defenders
        (-30, -10), (-30, 0), (-30, 10),
        # Five midfielders
        (-20, -23), (-20, -10), (-20, 0), (-20, 10), (-20, 23),
        # Two forwards
        (-10, -5), (-10, 5),
    ],
    "3-1-6": [
        # Three defenders
        (-30, -10), (-30, 0), (-30, 10),
        # One midfielder
        (-20, 0),
        # Six forwards
        (-10, -25), (-10, -15), (-10, -5), (-10, 5), (-10, 15), (-10, 25),
    ],
    "4-4-2": [
        # Four defenders
        (-27, -15), (-30, -5), (-30, 5), (-27, 15),
        # Four midfielders
        (-18, -15), (-20, -5), (-20, 5), (-18, 15),
        # Two forwards
        (-10, -5), (-10, 5),
    ],
    "4-3-3": [
        # Four defenders
        (-27, -15), (-30, -5), (-30, 5), (-27, 15),
        # Three midfielders
        (-20, -10), (-22, 0), (-20, 10),
        # Three forwards
        (-12, -12), (-10, 0), (-12, 12),
    ],
    "4-2-4": [
        # Four defenders
        (-27, -15), (-30, -5), (-30, 5), (-27, 15),
        # Two midfielders
        (-20, -5), (-20, 5),
        # Four forwards
        (-9, -15), (-9, -5), (-9, 5), (-9, 15),
    ],
    "4-5-1": [
        # Four defenders
        (-29, -15), (-30, -5), (-30, 5), (-29, 15),
        # Five midfielders
        (-17, -20), (-20, -10), (-20, 0), (-20, 10), (-17, 20),
        # One forward
        (-12, 0),
    ],
    "4-4-1-1": [
        # Four defenders
        (-30, -15), (-30, -5), (-30, 5), (-30, 15),
        # Four midfielders
        (-20, -15), (-20, -5), (-20, 5), (-20, 15),
        # One midfielder
        (-10, 0),
        # One forward
        (-5, 0),
    ],
    "5-3-2": [
        # Five defenders
        (-25, -23), (-30, -10), (-30, 0), (-30, 10), (-25, 23),
        # Three midfielders
        (-20, -10), (-20, 0), (-20, 10),
        # Two forwards
        (-10, -5), (-10, 5),
    ],
    "5-2-3": [
        # Five defenders
        (-30, -23), (-30, -10), (-30, 0), (-30, 10), (-30, 23),
        # Two midfielders
        (-20, -5), (-20, 5),
        # Three forwards
        (-10, -10), (-10, 0), (-10, 10),
    ],
}


def plot_formation(formation, figax=None, team_color='k'):
    """
    Plot the formation templates
    """
    if figax is None: 
        fig, ax = mviz.plot_pitch()
    else: 
        fig, ax = figax
    x_positions = [pos[0] for pos in formation]
    y_positions = [pos[1] for pos in formation]
    ax.scatter(x_positions, y_positions, color=team_color, s = 120)
    plt.show()

for formation_name, formation in formations.items():
    print(formation_name)
    plot_formation(formation)







def scale_translate_rotate(source, target, max_rotation_degrees=5):
    """
    Function to scale, translate, and rotate the formation templates
    """
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    source_scale = np.mean(np.linalg.norm(source_centered, axis=1))
    target_scale = np.mean(np.linalg.norm(target_centered, axis=1))
    scale_factor = target_scale / source_scale
    source_scaled = source_centered * scale_factor
    H = source_scaled.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    rotation_matrix = Vt.T @ U.T
    rotation_angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1))
    max_rotation_angle = np.deg2rad(max_rotation_degrees)
    if rotation_angle > max_rotation_angle:
        rotation_angle = max_rotation_angle
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)]])
    source_rotated = source_scaled @ rotation_matrix
    source_transformed = source_rotated + target_centroid
    return source_transformed


def calculate_distance(source, target):
    """
    Function to calculate the distance between player positions and templates
    """
    hausdorff_dist = max(directed_hausdorff(source, target)[0], directed_hausdorff(target, source)[0])
    return hausdorff_dist

def extract_positions(df, frame, keeper, home=True):
    """
    Function to extract the player positions from each frame
    """
    prefix = 'Home' if home else 'Away'
    keeper_number = int(keeper)  # parse keeper into integer
    positions = []

    # Select all columns that start with the team prefix and end with '_x' or '_y'
    x_columns = df.filter(regex=f'^{prefix}_.*_x$').columns
    y_columns = df.filter(regex=f'^{prefix}_.*_y$').columns

    for x_col, y_col in zip(x_columns, y_columns):
        if 'ball' in x_col or 'ball' in y_col:
            continue
        if pd.isna(df.loc[frame, x_col]) or pd.isna(df.loc[frame, y_col]):  # skip NaN values
            continue
        if f'{prefix}_{keeper_number}_x' == x_col or f'{prefix}_{keeper_number}_y' == y_col:  # skip keeper
            continue
        positions.append([df.loc[frame, x_col], df.loc[frame, y_col]])
    return positions


def find_best_match(observation, formations, max_rotation_degrees=5, plot=False):
    """
    Function to find the best matched formation between players and adapted templates
    """
    best_match = None
    min_distance = np.inf
    for name, formation in formations.items():
        # Skip if the number of positions doesn't match
        if len(formation) != len(observation):
            continue
        source_transformed = scale_translate_rotate(np.array(formation), np.array(observation), max_rotation_degrees)
        distance = calculate_distance(source_transformed, np.array(observation))
        if distance < min_distance:
            min_distance = distance
            best_match = name
            best_matcH_transformed = source_transformed
    # if plot and best_match is not None:
        # plt.scatter(*zip(*observation), color='blue')
        # plt.scatter(*zip(*best_matcH_transformed), color='red')
        # plt.show()
    return best_match

def analyze_sequences(tracking_data, sequences_df, formations, keeper='1', home=True):
    """
    Function to use the previous functions over a set of sequences and return counts
    for the percentage of time a team spends best matched with a certain formation
    """
    formation_columns = [f"formation_{name}" for name in formations.keys()]
    for col in formation_columns:
        sequences_df[col] = 0

    total_frames_all_sequences = 0
    total_formation_frames_all_sequences = {name: 0 for name in formations.keys()}

    for i, row in sequences_df.iterrows():
        start_frame, end_frame = int(row['start_frame']), int(row['end_frame'])
        total_frames = (end_frame - start_frame) // 10 + 1
        total_frames_all_sequences += total_frames

        formation_counts = {name: 0 for name in formations.keys()}

        for frame in range(start_frame, end_frame+1, 10):
            positions = extract_positions(tracking_data, frame, keeper, home)
            best_match = find_best_match(positions, formations)
            formation_counts[best_match] += 1

        for name, count in formation_counts.items():
            sequences_df.loc[i, f"formation_{name}"] = count / total_frames
            total_formation_frames_all_sequences[name] += count

    # Create a new DataFrame to store total percentages
    total_percentages = {f"formation_{name}": count / total_frames_all_sequences 
                         for name, count in total_formation_frames_all_sequences.items()}
    total_percentages_df = pd.DataFrame(total_percentages, index=['Total'])
    return total_percentages_df






def extract_data_for_sequence(tracking_data, start_frame, end_frame, sample_rate=1):
    """
    Extracts data for a given sequence based on start and end frame.
    """
    return tracking_data.loc[start_frame:end_frame:sample_rate]

def compute_hull_for_frame(frame_row, frame_number, team='Home', keeper_number=None):
    """
    Computes the convex hull for a single frame and returns its unique vertices.
    """
    all_columns_x = [col for col in frame_row.index if (team in col) and ('_x' in col) and ('ball' not in col)]
    all_columns_y = [col for col in frame_row.index if (team in col) and ('_y' in col) and ('ball' not in col)]
    
    valid_points = [(frame_row[col_x], frame_row[col_y]) for col_x, col_y in zip(all_columns_x, all_columns_y) 
                    if not np.isnan(frame_row[col_x]) and str(keeper_number) not in col_x]
    if len(valid_points) < 3:
        return []
    hull = ConvexHull(valid_points)
    # Extract unique vertices
    unique_vertices = [np.array(valid_points)[index] for index in hull.vertices]
    return unique_vertices

def compute_hull_for_sequence(sequence_data, team='Home', keeper_number=None):
    """
    Computes the vertices of the convex hulls for each frame in the sequence.
    """
    vertices_list = []
    for frame_number, (_, frame_row) in enumerate(sequence_data.iterrows()):
        vertices = compute_hull_for_frame(frame_row, frame_number, team, keeper_number)
        vertices_list.extend(vertices)
    return vertices_list

def perform_mini_batch_kmeans(all_vertices, n_clusters=6):
    """
    Performs Mini-Batch KMeans clustering on the given vertices.
    """
    # Convert vertices to numpy array
    all_vertices = np.array(all_vertices)
    
    # Ensure that number of clusters doesn't exceed number of data points
    n_clusters = min(n_clusters, len(all_vertices))
    
    # If no valid data points, return an empty list
    if n_clusters == 0:
        print("Warning: No valid data points found for clustering.")
        return pd.DataFrame()
    # Create and fit the Mini-Batch KMeans model
    mini_batch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit(all_vertices)
    # Corrected this line to use mini_batch_kmeans.cluster_centers_
    return pd.DataFrame(mini_batch_kmeans.cluster_centers_, columns=['X', 'Y'])

def plot_clusters(cluster_centers, points, title=None):
    """
    Plots the given clusters and points.
    """
    if len(points) == 0 or len(cluster_centers) == 0:
        print("Warning: No data points or cluster centers to plot.")
        return
    
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1], c='lightgrey', marker='o', label='Data Points')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=180, label='Cluster Centers')
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    if title:
        plt.title(f'Mini-Batch KMeans Clustering {title}', fontsize=17)
        plt.savefig(fname=title + '.svg', format='svg', dpi=400, bbox_inches='tight', pad_inches=0)
    else:
        plt.title('Mini-Batch KMeans Clustering', fontsize=17)
    plt.show()


def main(tracking_data, sequences, sample_rate=1, team='Home', keeper_number=None, n_clusters=7, title=None):
    """
    Executes all the functions involved with clustering to save the points
    of the average vertices for the entire sequence.
    """
    all_vertices = []
    for _, row in sequences.iterrows():
        sequence_data = extract_data_for_sequence(tracking_data, row['start_frame'], row['end_frame'], sample_rate)
        vertices = compute_hull_for_sequence(sequence_data, team, keeper_number)
        all_vertices.extend(vertices)
    cluster_centers_df = perform_mini_batch_kmeans(all_vertices, n_clusters)
    plot_clusters(cluster_centers_df.values, all_vertices, title)  # Use the DataFrame values for plotting
    return cluster_centers_df









def top_players_for_events(events_df):
    """
    Get the 3 players with the top event type values
    """
    # Define the events of interest
    events_of_interest = ['PASS', 'SHOT', 'CHALLENGE', 'RECOVERY']
    results = {}
    for event_type in events_of_interest:
        # Filter by event type
        event_data = events_df[events_df['Type'] == event_type]
        # Count occurrences for each player
        player_counts = event_data['From'].value_counts().head(3)
        # Store the top 3 players and their counts for the current event type
        results[event_type] = {
            'players': player_counts.index.tolist(),
            'counts': player_counts.values.tolist()
        }
    return results








def team_intensity(tracking_data, team_prefix, sequences, sprinting_weight=3):
    """
    Get a value for the amount of sprinting and running a team does per
    minute, to see the intenisty of a team within each phase.
    """
    total_intensity = 0
    total_running_distance_rate = 0 # Meters per minute
    total_sprinting_distance_rate = 0 # Meters per minute
    total_time = 0

    for index, row in sequences.iterrows():
        start_frame, end_frame = row['start_frame'], row['end_frame']
        sequence_time = (end_frame - start_frame + 1) / 25 / 60 # Time in minutes for this sequence
        total_time += sequence_time
        running_distance = 0
        sprinting_distance = 0
        for column in tracking_data.columns:
            if team_prefix in column and '_speed' in column:
                running_distance += tracking_data.loc[(tracking_data[column] >= 4) & (tracking_data[column] < 7) & (tracking_data.index >= start_frame) & (tracking_data.index <= end_frame), column].sum() / 25
                sprinting_distance += tracking_data.loc[(tracking_data[column] >= 7) & (tracking_data.index >= start_frame) & (tracking_data.index <= end_frame), column].sum() / 25

        sequence_intensity = (running_distance + sprinting_distance) / sequence_time
        total_intensity += sequence_intensity
        total_running_distance_rate += running_distance / sequence_time
        total_sprinting_distance_rate += sprinting_distance / sequence_time

    if total_time == 0:
        return pd.DataFrame()

    average_intensity = total_intensity / len(sequences)
    average_running_distance_rate = total_running_distance_rate / len(sequences)
    average_sprinting_distance_rate = total_sprinting_distance_rate / len(sequences)
    weighted_intensity = (average_running_distance_rate + sprinting_weight * 30 * average_sprinting_distance_rate)/30

    return pd.DataFrame({
        'Team': [team_prefix],
        'Intensity [m/min]': [average_intensity],
        'Running Distance Rate [m/min]': [average_running_distance_rate],
        'Sprinting Distance Rate [m/min]': [average_sprinting_distance_rate],
        'Weighted Intesity Score': [round(weighted_intensity/10)]
    })








def man_press_distance(H_tracking, A_tracking, sequences, non_possession_team_prefix, frame_to_return=None):
    """
    Computes the total distance between the closest 4,6 or 8 players in the possession
    team to the 4,6 or 8 players in the possession team, then create a score based on
    these values to measure the level of man pressing a team uses.
    """
    total_distance_8 = total_distance_6 = total_distance_4 = 0
    total_frames = 0
    
    plot_data = None
    
    for index, row in sequences.iterrows():
        start_frame, end_frame = row['start_frame'], row['end_frame']
        for frame in range(int(start_frame), int(end_frame) + 1, 10): # Step size of 20
            if non_possession_team_prefix == 'Home':
                non_possession_data = H_tracking.loc[frame]
                possession_data = A_tracking.loc[frame]
            else:
                non_possession_data = A_tracking.loc[frame]
                possession_data = H_tracking.loc[frame]
                
            total_frames += 1
            non_possession_columns = [col for col in non_possession_data.index if f'{non_possession_team_prefix}_' in col and ('_x' in col or '_y' in col) and 'ball' not in col and not pd.isnull(non_possession_data[col])]
            possession_columns = [col for col in possession_data.index if f'{"Away" if non_possession_team_prefix == "Home" else "Home"}_' in col and ('_x' in col or '_y' in col) and 'ball' not in col and not pd.isnull(possession_data[col])]
            
            non_possession_positions = [(non_possession_data[non_possession_columns[i]], non_possession_data[non_possession_columns[i + 1]]) for i in range(0, len(non_possession_columns), 2)]
            possession_positions = [(possession_data[possession_columns[j]], possession_data[possession_columns[j + 1]]) for j in range(0, len(possession_columns), 2)]
            distances = cdist(non_possession_positions, possession_positions, metric='euclidean')
            sorted_indices = np.argsort(distances, axis=None)
            unique_non_possession_indices = set()
            unique_possession_indices = set()
            unique_distances = []
            
            for index in sorted_indices:
                i, j = np.unravel_index(index, distances.shape)
                if i not in unique_non_possession_indices and j not in unique_possession_indices:
                    unique_distances.append((i, j, distances[i, j]))
                    unique_non_possession_indices.add(i)
                    unique_possession_indices.add(j)

            sorted_unique_distances = sorted(unique_distances, key=lambda x: x[2])
            total_distance_8 += sum(x[2] for x in sorted_unique_distances[:8])
            total_distance_6 += sum(x[2] for x in sorted_unique_distances[:6])
            total_distance_4 += sum(x[2] for x in sorted_unique_distances[:4])
            if frame_to_return and frame == frame_to_return:
                plot_data = (non_possession_positions, possession_positions, sorted_unique_distances)
    
    average_distance_8 = total_distance_8 / total_frames if total_frames > 0 else 0
    average_distance_6 = total_distance_6 / total_frames if total_frames > 0 else 0
    average_distance_4 = total_distance_4 / total_frames if total_frames > 0 else 0
    weighted_average_distance = (2/9 * average_distance_8) + (5/9 * average_distance_6) + (2/9 * average_distance_4) 
    return {
        'result': pd.DataFrame({
            'Average Total Distance for 8 Players [m]': [average_distance_8],
            'Average Total Distance for 6 Players [m]': [average_distance_6],
            'Average Total Distance for 4 Players [m]': [average_distance_4],
            'Weighted Man Pressing Score': [round(10*weighted_average_distance)/10]
            }),
        'plot_data': plot_data
    }


    





def avg_phase_length(sequences_df, events_df):
    """
    Computes the average length of phases, both in passes and in time [s]
    """
    total_time_length = 0
    total_passes_length = 0

    for index, row in sequences_df.iterrows():
        start_frame, end_frame = row['start_frame'], row['end_frame']
        # Convert frame numbers to time if you have frame rate information (e.g. 25 frames per second)
        phase_time_length = (end_frame - start_frame) / 25  # Change 25 to your frame rate
        total_time_length += phase_time_length
        # Calculate the number of passes in the sequence
        passes_in_phase = events_df[(events_df['Start Frame'] >= start_frame) & (events_df['End Frame'] <= end_frame) & (events_df['Type'] == 'PASS')]
        phase_passes_length = len(passes_in_phase)
        total_passes_length += phase_passes_length

    avg_time_length = total_time_length / len(sequences_df) if len(sequences_df) > 0 else 0
    avg_passes_length = total_passes_length / len(sequences_df) if len(sequences_df) > 0 else 0
    return pd.DataFrame({'Average Phase Time [s]': [avg_time_length], 'Average Phase Passes': [avg_passes_length]})








def common_pass_dist(sequences_df, events_df):
    """
    Gets counts for the number of each length pass used per phase, then
    apply these counts to a score value shows how direct a team are.
    """
    # Counts for different distance categories
    distance_counts = {
        '0-10': 0,
        '10-20': 0,
        '20-35': 0,
        '35+': 0
    }
    num_sequences = len(sequences_df)
    # Iterate through the sequences
    for _, row in sequences_df.iterrows():
        start_frame, end_frame = row['start_frame'], row['end_frame']
        # Filter the events for passes within the sequence
        passes = events_df[(events_df['Start Frame'] >= start_frame) & 
                           (events_df['End Frame'] <= end_frame) & 
                           (events_df['Type'] == 'PASS')]
        # Iterate through the passes and calculate the distance
        for _, pass_row in passes.iterrows():
            start_x, start_y = pass_row['Start X'], pass_row['Start Y']
            end_x, end_y = pass_row['End X'], pass_row['End Y']
            pass_length = euclidean((start_x, start_y), (end_x, end_y))
            # Categorize the pass by length
            if pass_length < 10:
                distance_counts['0-10'] += 1
            elif pass_length < 20:
                distance_counts['10-20'] += 1
            elif pass_length < 35:
                distance_counts['20-35'] += 1
            else:
                distance_counts['35+'] += 1
    # Normalize the distance counts by dividing each value by num_sequences
    for key in distance_counts.keys():
        distance_counts[key] /= num_sequences
    weighted_score = 20 + (-10 * distance_counts['0-10'] + 
                      -2 * distance_counts['10-20'] + 
                      6 * distance_counts['20-35'] + 
                      35 * distance_counts['35+'])
    distance_counts['Score'] = round(weighted_score*10) /10  
    # Convert the counts into a DataFrame
    result_df = pd.DataFrame([distance_counts], columns=['0-10', '10-20', '20-35', '35+', 'Score'])
    return result_df







def flip_formations(formations):
    """
    Flip the formation templates to suit the home team playing
    right to left.
    """
    flipped_formations = {}
    for name, formation in formations.items():
        flipped_formations[name] = [(-x, y) for x, y in formation]
    return flipped_formations








# Call the main function for the creation of average convex hulls across a sequence
H_build_convex = main(H_tracking, H_build_ups, sample_rate=10, team='Home', keeper_number=H_keeper, n_clusters=8, title = 'for Team A Build Ups')
H_possess_convex = main(H_tracking, H_possess, sample_rate=10, team='Home', keeper_number=H_keeper, n_clusters=8)
H_defend_convex = main(H_tracking, A_possess, sample_rate=10, team='Home', keeper_number=H_keeper, n_clusters=8)
H_press_convex = main(H_tracking, A_build_ups, sample_rate=10, team='Home', keeper_number=H_keeper, n_clusters=8)
A_build_convex = main(A_tracking, A_build_ups, sample_rate=10, team='Away', keeper_number=A_keeper, n_clusters=8)
A_possess_convex = main(A_tracking, A_possess, sample_rate=10, team='Away', keeper_number=A_keeper, n_clusters=8)
A_defend_convex = main(A_tracking, H_possess, sample_rate=10, team='Away', keeper_number=A_keeper, n_clusters=8)
A_press_convex = main(A_tracking, H_build_ups, sample_rate=10, team='Away', keeper_number=A_keeper, n_clusters=8)



# Percentage best matched fomrations for home playing direction
flipped_formations = flip_formations(formations)
H_build_shape = analyze_sequences(H_tracking, H_build_ups, flipped_formations, H_keeper, 'Home')
H_press_shape = analyze_sequences(H_tracking, A_build_ups, flipped_formations, H_keeper, 'Home')
H_possess_shape = analyze_sequences(H_tracking, H_possess, flipped_formations, H_keeper, 'Home')
H_defend_shape = analyze_sequences(H_tracking, A_possess, flipped_formations, H_keeper, 'Home')
# Percentage best matched fomrations for away team
A_build_shape = analyze_sequences(A_tracking, A_build_ups, formations, A_keeper, False)
A_press_shape = analyze_sequences(A_tracking, H_build_ups, formations, A_keeper, False)
A_possess_shape = analyze_sequences(A_tracking, A_possess, formations, A_keeper, False)
A_defend_shape = analyze_sequences(A_tracking, H_possess, formations, A_keeper, False)


# The players with highest event counts for each team
A_top_player = top_players_for_events(A_events)
H_top_player = top_players_for_events(H_events)


# A measure of directness for possession and build-ups, using pass length
H_build_pass_length = common_pass_dist(H_build_ups, H_events)
A_build_pass_length = common_pass_dist(A_build_ups, A_events)
H_possess_pass_length = common_pass_dist(H_possess, H_events)
A_possess_pass_length = common_pass_dist(A_possess, A_events)


# A measure of directness for teams, using the length of phases in time[s] and passes
A_direct_build = avg_phase_length(A_build_ups, A_events)
H_direct_build = avg_phase_length(H_build_ups, H_events)
A_direct_possess = avg_phase_length(A_possess, A_events)
H_direct_possess = avg_phase_length(H_possess, H_events)


# The level of intensity each team plays at within each phase, measured
# by their amount of running and sprinting per minute of each phase
H_press_intensity = team_intensity(H_tracking, 'Home', A_build_ups)
A_press_intensity = team_intensity(A_tracking, 'Away', H_build_ups)
H_defend_intensity = team_intensity(H_tracking, 'Home', A_possess)
A_defend_intensity = team_intensity(A_tracking, 'Away', H_possess)
H_build_intensity = team_intensity(H_tracking, 'Home', H_build_ups)
A_build_intensity = team_intensity(A_tracking, 'Away', A_build_ups)
H_possess_intensity = team_intensity(H_tracking, 'Home', H_possess)
A_possess_intensity = team_intensity(A_tracking, 'Away', A_possess)


# Man pressing ratings for pressing and for defence
A_man_press = man_press_distance(H_tracking, A_tracking, H_build_ups, 'Away')['result']
H_man_press = man_press_distance(H_tracking, A_tracking, A_build_ups, 'Home')['result']
A_man_defend = man_press_distance(H_tracking, A_tracking, H_possess, 'Away')['result']
H_man_defend = man_press_distance(H_tracking, A_tracking, A_possess, 'Home')['result']


