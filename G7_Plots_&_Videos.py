"""
This file plots all of the visual information which has been extracted
during this analysis, GK passing, pesses before shots, most common
formations, and the average convex hull for the phasepresented underneath
the average positions of the player throughout that phase. In contrast to other
files the calling of cuntions directly follows that function.
"""



from D4_Data_Entry_Summary import team_coloring
import matplotlib
from moviepy.editor import VideoFileClip
import os

matplotlib.rcParams['font.serif'] = ["Times New Roman"] 





def plot_best_match_formation(tracking_data, formations, frame, keeper, home=True):
    """
    Function to plot the adapted formation template with a given frame of
    player positions.
    """
    # Extract real positions
    real_positions = extract_positions(tracking_data, frame, keeper, home)
    
    # Find the best match formation and get the transformed positions
    best_match_name = find_best_match(real_positions, formations)
    best_match_formation = formations[best_match_name]
    transformed_positions = scale_translate_rotate(np.array(best_match_formation), np.array(real_positions))

    # Plot real positions
    real_x, real_y = zip(*real_positions)
    plt.scatter(real_x, real_y, color=team_coloring[0], label='Real Positions', s=100)
    fonsize=15
    # Plot transformed positions
    transformed_x, transformed_y = zip(*transformed_positions)
    plt.scatter(transformed_x, transformed_y, color='k', label=f'Best Match: {best_match_name}', s=100)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=fonsize)
    plt.xlabel('X', fontsize=fonsize)
    plt.ylabel('Y', fontsize=fonsize)
    
    # Corrected title assignment
    plot_title = f'Frame {frame} - Real vs Best Match Formation for Team A'
    plt.title(plot_title, fontsize=fonsize+2)
    
    plt.savefig(fname=plot_title + '.svg', format='svg', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

# Plot for Home team, this frame is used in the project and only works for game 1
plot_best_match_formation(H_tracking, flipped_formations, 59421, H_keeper_number, home=True)







def plot_connections(plot_data, title):
    """
    Function to plot the connections between 8, 6 and 4 players
    on the non-possession team and their closest player in the 
    possession team, for a given frame, shwoing man-to-man marking.
    """
    non_possession_positions, possession_positions, sorted_unique_distances = plot_data
    cases = [8, 6, 4]
    titles = ['8 Players', '6 Players', '4 Players']

    fig, axs = plt.subplots(1, 3, figsize=(12,5))  # Adjust figsize as needed

    for idx, (num_connections, subplot_title) in enumerate(zip(cases, titles)):
        ax = axs[idx]
        ax.scatter(*zip(*non_possession_positions), label='Non-possession team' if idx == 0 else "", c='b')  # Change color as needed
        ax.scatter(*zip(*possession_positions), label='Possession team' if idx == 0 else "", c='r')  # Change color as needed

        for i, j, _ in sorted_unique_distances[:num_connections]:
            ax.plot([non_possession_positions[i][0], possession_positions[j][0]], 
                    [non_possession_positions[i][1], possession_positions[j][1]], 'k-')

        ax.set_title(subplot_title, fontsize=17)
        ax.set_xlabel('X', fontsize=15)
        if idx == 0:
            ax.set_ylabel('Y', fontsize=15)
        ax.set_xlim(-35, 10)

    # Setting the main title for all subplots
    fig.suptitle(title, fontsize=20)

    # Creating a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{title}.svg", format='svg', dpi=200, bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.show()

# Can only print start_frame + multiples of 10
frame_number = 55497 # the frame number you want to plot
result_data = man_press_distance(H_tracking, A_tracking, A_build_up_sequences, 'Home', frame_to_return=frame_number)
if result_data['plot_data']:
    plot_connections(result_data['plot_data'], title = 'Team A Man pressing - Frame 59497')







def plot_avg_positions_all_sequences(sequences, H_tracking, A_tracking, H_summary, A_summary, H_hull=None, A_hull=None, team='both', filen = None):
    """
    Function to plot the average positions of players over a set of sequences
    defined as a phase of play that we want to analyse. It plots the average
    positions of the 11 most seen players across this phase and then uses the
    K-means cluster centres to add an average convex hull underneath.
    """
    # Sort players based on the 'Minutes Played' and get the top 11 players
    H_players = H_summary.sort_values(by='Minutes Played', ascending=False).index[:11].tolist() 
    A_players = A_summary.sort_values(by='Minutes Played', ascending=False).index[:11].tolist()

    # Append player coordinates ('_x' and '_y') to the list of players
    H_players = ['Home_'+player+'_x' for player in H_players] + ['Home_'+player+'_y' for player in H_players]
    A_players = ['Away_'+player+'_x' for player in A_players] + ['Away_'+player+'_y' for player in A_players]

    # Append 'ball_x' and 'ball_y' to the list of players to include the ball
    H_players += ['ball_x', 'ball_y']
    A_players += ['ball_x', 'ball_y']

    # Initialize lists to store all the positions
    all_positions_home = []
    all_positions_away = []

    # Iterate over the sequences
    for _, row in sequences.iterrows():
        start_frame = row['start_frame']
        end_frame = row['end_frame']

        # Compute average positions for home team and store
        avg_home = H_tracking.loc[start_frame:end_frame, H_players].mean()
        all_positions_home.append(avg_home)

        # Compute average positions for away team and store
        avg_away = A_tracking.loc[start_frame:end_frame, A_players].mean()
        all_positions_away.append(avg_away)
        
    # Convert the lists of average positions to DataFrames
    avg_positions_home = pd.DataFrame(all_positions_home).mean()
    avg_positions_away = pd.DataFrame(all_positions_away).mean()
    # Convert the average positions to single-row DataFrames
    avg_positions_home = pd.DataFrame([avg_positions_home.to_dict()])
    avg_positions_away = pd.DataFrame([avg_positions_away.to_dict()])

     # Create DataFrames with NaNs for missing teams
    empty_home = pd.DataFrame([dict.fromkeys(avg_positions_home.columns, np.nan)])
    empty_away = pd.DataFrame([dict.fromkeys(avg_positions_away.columns, np.nan)])
        
    def plot_hull(ax, hull_df, color='lightgrey'):
        """
        Function to plot the average convex hull from the cluster centres
        """
        if hull_df is not None and not hull_df.empty:
            # Create a convex hull from the dataframe
            points = hull_df[['X', 'Y']].values
            if len(points) >= 3:
                hull = ConvexHull(points)
                polygon_points = np.append(hull.points[hull.vertices], [hull.points[hull.vertices][0]], axis=0)
                ax.plot(polygon_points[:, 0], polygon_points[:, 1], c=color, alpha=0.3)
                ax.fill(polygon_points[:, 0], polygon_points[:, 1], alpha=0.3, fc=color, ec=color)
                
                hull_area = hull.volume
                print(f"Area of the convex hull: {hull_area:.2f} square units")

    # Plot the average positions:
    if team == 'home':
        fig, ax = mviz.plot_frame(avg_positions_home.iloc[0], empty_away.iloc[0], Playermarkersize=34, annotate=True, num_move_x = 1.65, num_move_y = 1.2)
        plot_hull(ax, H_hull)
        
        # Reflect the plot along the x=0 axis only for team='home'
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])
        
    elif team == 'away':
        fig, ax = mviz.plot_frame(empty_home.iloc[0], avg_positions_away.iloc[0], Playermarkersize=34, annotate=True, num_move_x = -1.65, num_move_y = -1.2)
        plot_hull(ax, A_hull)
    else:
        raise ValueError("Invalid value for 'team'. Choose 'home' or 'away'.")
    filename = f"{filen}.svg"
    
    ax.set_position([0, 0, 1, 1])
    plt.savefig(filename, format='svg', dpi=80, bbox_inches='tight', pad_inches=0)
    plt.show()

plot_avg_positions_all_sequences(A_build_up_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=H_press_convex, A_hull=None, team='home', filen = "H_press")
plot_avg_positions_all_sequences(A_build_up_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=None, A_hull=A_build_convex, team='away', filen = "A_build")
plot_avg_positions_all_sequences(H_build_up_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=H_build_convex, A_hull=None, team='home', filen = "H_build")
plot_avg_positions_all_sequences(H_build_up_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=None, A_hull=A_press_convex, team='away', filen = "A_press")
plot_avg_positions_all_sequences(A_possess_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=H_defend_convex, A_hull=None, team='home', filen = "H_defend")
plot_avg_positions_all_sequences(A_possess_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=None, A_hull=A_possess_convex, team='away', filen = "A_posses")
plot_avg_positions_all_sequences(H_possess_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=H_possess_convex, A_hull=None, team='home', filen = "H_possess")
plot_avg_positions_all_sequences(H_possess_sequences, H_tracking, A_tracking, H_physical_summary, A_physical_summary, H_hull=None, A_hull=A_defend_convex, team='away', filen = "A_defend")







# calculate total goals, shots, passes, total running distance, and total sprints for each team
def calculate_totals(events_df, summary_df):
    """
    Function to create and store totals of general match stats.
    """
    total_goals = events_df[(events_df['Subtype'].str.contains('GOAL')) & (events_df['Type'] == 'SHOT')]['Type'].count()
    total_shots = events_df[events_df['Type'] == 'SHOT']['Type'].count()
    total_passes = events_df[events_df['Type'] == 'PASS']['Type'].count()
    total_distance = summary_df['Distance [km]'].sum()
    total_sprints = summary_df['# sprints'].sum()
    
    return [total_goals, total_shots, total_passes, total_distance, total_sprints]

H_totals = calculate_totals(H_events, H_physical_summary)
A_totals = calculate_totals(A_events, A_physical_summary)






def plot_all_passes(team_events, title, H_or_A):
    """
    Function to plot a dataframe of events which includes just passes.
    """
    fig, ax = mviz.plot_pitch()
    for i, team_event in team_events.iterrows():
        if H_or_A == 'home':
            ax.arrow(team_event['Start X'], team_event['Start Y'], team_event['End X'] - team_event['Start X'], team_event['End Y'] - team_event['Start Y'],
                     head_width=2.5, head_length=2.5, linewidth=2.5, fc= team_coloring[0], ec= team_coloring[0])
        else:
            ax.arrow(team_event['Start X'], team_event['Start Y'], team_event['End X'] - team_event['Start X'], team_event['End Y'] - team_event['Start Y'],
                     head_width=2.5, head_length=2.5, linewidth=2.5, fc= team_coloring[1], ec=team_coloring[1])
    
    if H_or_A == 'home':
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])
        
    ax.set_position([0, 0, 1, 1])
    plt.savefig(f"{title}.svg", format='svg', dpi=80) # Save the figure
    plt.show() # Show the figure
    
# Plot all the goalkeeper passes using plot_all_passes
plot_all_passes(H_gk_pass_events, f"All home GK Passes (Game {game_id}, Team A)", 'home')
plot_all_passes(A_gk_pass_events, f"All away GK Passes (Game {game_id}, Team B)", 'away')








def plot_passes_and_shots(passes, shots, title, H_or_A):
    """
    Function to plot a dataframe of events which include passes and shots.
    """
    fig, ax = mviz.plot_pitch()
    
    # Plot passes
    for i, pass_event in passes.iterrows():
        if H_or_A == 'home':
            ax.arrow(pass_event['Start X'], pass_event['Start Y'], pass_event['End X'] - pass_event['Start X'], pass_event['End Y'] - pass_event['Start Y'],
                     head_width=2.5, head_length=2.5, linewidth=2.5, fc=team_coloring[0], ec=team_coloring[0])
        else:
            ax.arrow(pass_event['Start X'], pass_event['Start Y'], pass_event['End X'] - pass_event['Start X'], pass_event['End Y'] - pass_event['Start Y'],
                     head_width=2.5, head_length=2.5, linewidth=2.5, fc=team_coloring[1], ec=team_coloring[1])
    
    # Plot shots
    ax.scatter(shots['Start X'], shots['Start Y'], color='black', s=70, label='Shots')
    
    # Reflect the plot along the x=0 axis for 'home'
    if H_or_A == 'home':
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])
    
    ax.set_position([0, 0, 1, 1])
    plt.savefig(f"{title}.svg", format='svg', dpi=80, bbox_inches='tight', pad_inches=0) # Save the figure
    plt.show()

plot_passes_and_shots(H_pre_shot_pass_events, H_shots_events, f"All home passes to shots (Game {game_id}, Team A)", 'home')
plot_passes_and_shots(A_pre_shot_pass_events, A_shots_events, f"All away passes to shots (Game {game_id}, Team B)", 'away')







def plot_top_formations(percent_df, formations, title=None):
    """
    Function to plot the top 2 formations as located by the shape
    analysis, the shape and its percentage is plotted.
    """
    # Transpose the dataframe, reset the index, and rename columns
    percent_df = percent_df.transpose().reset_index()
    percent_df.columns = ['formation', 'percentage']
    
    # Sort the dataframe to get top 2 formations
    top_formations = percent_df.sort_values(by='percentage', ascending=False).head(2)
    
    # Create a figure with two subplots (side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.8)
    
    # Plot each formation in its respective subplot
    for i, (index, row) in enumerate(top_formations.iterrows()):
        ax = axes[i]
        
        # Extract the numbers from the formation name
        formation_name = '-'.join(filter(str.isdigit, row['formation']))
        
        formation_data = formations[formation_name]
        
        # Scaling the coordinates to make the formation 1/3 smaller and shifting higher
        scaled_data = [(x * 1/3, y * 0.2/3) for x, y in formation_data]
        
        for x, y in scaled_data:
            ax.scatter(x, y, s=800, c='black', edgecolors='black')  # Set color to black
        
        # Annotate with formation name and percentage, multiplied by 100
        percentage = row['percentage'] * 100
        ax.text(0.5, -0.15, f"{formation_name} : {int(percentage)}%", transform=ax.transAxes, ha='center', fontsize=60, fontweight='bold')
        # Remove the axis
        ax.axis('off')
    

    plt.savefig(f"{title}.svg", format='svg', dpi=80) # Save the figure
    plt.show()

# Example usage:
plot_top_formations(A_build_shape, formations, title = 'A_build_shapes')
plot_top_formations(A_possess_shape, formations, title = 'A_possess_shapes')
plot_top_formations(A_press_shape, formations, title = 'A_press_shapes')
plot_top_formations(A_defend_shape, formations, title = 'A_defend_shapes')
plot_top_formations(H_build_shape, formations, title = 'H_build_shapes')
plot_top_formations(H_possess_shape, formations, title = 'H_possess_shapes')
plot_top_formations(H_press_shape, formations, title = 'H_press_shapes')
plot_top_formations(H_defend_shape, formations, title = 'H_defend_shapes')





# Insert your own base path for the folders
DIR = '/Users/jamesgothard/fpath'

def create_highlight_reel(sequences_df, H_tracking, A_tracking, title="Title"):
    """
    Function to create a folder containing animated clips of each sequence
    within each phase.
    """
    # Base directory for the clips
    base_dir = DIR
    
    # Create a directory for this highlight set
    highlight_dir = os.path.join(base_dir, title.replace(" ", "_"))  # Replace spaces with underscores
    if not os.path.exists(highlight_dir):
        os.makedirs(highlight_dir)
    
    for _, row in sequences_df.iterrows():
        start_frame, end_frame = row['start_frame'], row['end_frame']
        
        hometeam_clip = H_tracking.loc[start_frame:end_frame]
        awayteam_clip = A_tracking.loc[start_frame:end_frame]
        
        # Define clip name and path
        clip_name = f'clip_{start_frame}_{end_frame}.mp4'
        full_clip_path = os.path.join(highlight_dir, clip_name)
        
        # Save each individual clip
        mviz.save_match_clip(hometeam_clip, awayteam_clip, highlight_dir, fname=clip_name[:-4], figax=None, frames_per_second=25, team_colors=('r','b'), field_dimen=(106.0,68.0), include_player_velocities=False, Playermarkersize=10, PlayerAlpha=0.7, title=title)

# create_highlight_reel(H_gk_pass_sequences, H_tracking, A_tracking, title="Home GK passes")
# create_highlight_reel(A_gk_pass_sequences, H_tracking, A_tracking, title="Away GK passes")
# create_highlight_reel(H_pre_shot_pass_sequences, H_tracking, A_tracking, title="Home passes before shots")
# create_highlight_reel(A_pre_shot_pass_sequences, H_tracking, A_tracking, title="Away passes before shots")
# create_highlight_reel(H_build_up_sequences, H_tracking, A_tracking, title="Home build-ups")
# create_highlight_reel(A_build_up_sequences, H_tracking, A_tracking, title="Away build-ups")
# create_highlight_reel(H_possess_sequences, H_tracking, A_tracking, title="Home possessions")
# create_highlight_reel(A_possess_sequences, H_tracking, A_tracking, title="Away possessions")






"""
Creating simple summary plots for each game in the dataset.
"""


data = {'home': H_totals, 'away': A_totals}
df = pd.DataFrame(data)
bar_width = 0.15
fsize=28
df.index = ['Goals', 'Shots', 'Passes', 'Total Distance [km]', 'Total Sprints']
fig, ax = plt.subplots(nrows=len(df.index), sharex=False, figsize=(8, 10))
fig.suptitle(f'Summary of Game {game_id}', fontsize=fsize + 2, fontname="Times New Roman")


for i in range(len(df.index)):
    indices = np.array([0])
    H_pos = indices + bar_width/1.8
    A_pos = indices - bar_width/1.8

    # draw the bars
    H_bars = ax[i].barh(H_pos, int(df.iloc[i]['home']), color=team_coloring[0], edgecolor='black', height=bar_width)
    A_bars = ax[i].barh(A_pos, int(df.iloc[i]['away']), color=team_coloring[1], edgecolor='black', height=bar_width)

    # label the bars
    ax[i].set_yticks(indices)
    ax[i].set_yticklabels([df.index[i]], fontname="Times New Roman", fontsize=fsize)
    
    # Set the x limit based on the maximum of home and away values
    ax[i].set_xlim([0, max(int(df.iloc[i]['home']), int(df.iloc[i]['away'])) * 1.1]) 

    # Write the values next to the bars
    ax[i].text(int(df.iloc[i]['home']), H_pos[0], " "+str(int(df.iloc[i]['home'])), color='black', va='center', fontname="Times New Roman", fontsize=fsize)
    ax[i].text(int(df.iloc[i]['away']), A_pos[0], " "+str(int(df.iloc[i]['away'])), color='black', va='center', fontname="Times New Roman", fontsize=fsize)

    # Remove the spines
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    
    # Remove the x-axis labels
    ax[i].set_xticks([])

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
filename = f"Summary_of_Game_{game_id}.svg"
plt.savefig(filename, format='svg', dpi=80)
plt.show()







