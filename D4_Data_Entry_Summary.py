"""
This section uses metrica functions to read in the data
and also does some data prep as well as choosing the game
and the colors to use in the analysis, H_ and A_ refer to
the Home and Away teams.
"""


import A1_Metrica_IO as mio
import B2_Metrica_Viz as mviz
import C3_Metrica_Velocities as mvel
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/jamesgothard/Research Project/Metrica/Python files')



# Write in the file path to where the Sample Data games are stored
DATADIR = '/Users/jamesgothard/Research Project/Metrica/sample-data-master/data'
# Choose the game to produce analysis on, game 1 or game 2
game_id = 1 
# Choose the colours for the home, then away team these colours are for game 1
team_coloring = ('r','b')
# For game 2, need to restart kernel to make colour change
# team_coloring = ('y','m')
# Pick the colour of some of the analysis plots
event_color = ('r')



# read in the event and tracking data
events = mio.read_event_data(DATADIR,game_id)
H_tracking = mio.tracking_data(DATADIR,game_id,'Home')
A_tracking = mio.tracking_data(DATADIR,game_id,'Away')




# Convert positions from metrica units to meters
H_tracking = mio.to_metric_coordinates(H_tracking)
A_tracking = mio.to_metric_coordinates(A_tracking)
events = mio.to_metric_coordinates(events)




# Find the playing direction of the home team
H_first_half_direction = mio.find_playing_direction(H_tracking, "Home")
# Make sure home play right to left, away left to right
if H_first_half_direction == 1:
    for team in [H_tracking, A_tracking, events]:
        columns = [c for c in team.columns if c[-1].lower() in ['x', 'y']]
        team.loc[:, columns] *= -1
H_tracking,A_tracking,events = mio.to_single_playing_direction(H_tracking,A_tracking,events)




# Get events by team
H_events = events[events['Team']=='Home']
A_events = events[events['Team']=='Away']
# Calculate player velocities
#H_tracking = mvel.calc_player_velocities(H_tracking,smoothing=True)
#A_tracking = mvel.calc_player_velocities(A_tracking,smoothing=True)
# **** NOTE *****
# if the lines above produce an error (happens for one version of numpy) change them to the lines below:
# ***************
H_tracking = mvel.calc_player_velocities(H_tracking,smoothing=True,filter_='moving_average')
A_tracking = mvel.calc_player_velocities(A_tracking,smoothing=True,filter_='moving_average')



"""
This section creates basic dataframes of shots, goals
and physical summaries of each team
"""




# Get all shots
H_shots = H_events[H_events.Type=='SHOT']
A_shots = A_events[A_events.Type=='SHOT']
# Get all goals
H_goals = H_shots[H_shots['Subtype'].str.contains('-GOAL')].copy()
A_goals = A_shots[A_shots['Subtype'].str.contains('-GOAL')].copy()
# Add a column event 'Minute' to the data frame
H_goals['Minute'] = H_goals['Start Time [s]']/60.
A_goals['Minute'] = A_goals['Start Time [s]']/60.





# Create summary of movement of all players
def create_summary(tracking_data, team_prefix):
    players = np.unique([c.split('_')[1] for c in tracking_data.columns if c[:4] == team_prefix])
    summary = pd.DataFrame(index=players)
    
    minutes_played = []
    for player in players:
        column = team_prefix + '_' + player + '_x'
        player_minutes = (tracking_data[column].last_valid_index() - tracking_data[column].first_valid_index() + 1) / 25 / 60
        minutes_played.append(player_minutes)
    summary['Minutes Played'] = minutes_played
    summary = summary.sort_values(['Minutes Played'], ascending=False)

    distance = []
    for player in summary.index:
        column = team_prefix + '_' + player + '_speed'
        player_distance = tracking_data[column].sum()/25./1000
        distance.append(player_distance)
    summary['Distance [km]'] = distance

    speed_bands = [('Walking [km]', 0, 2), ('Jogging [km]', 2, 4), ('Running [km]', 4, 7), ('Sprinting [km]', 7, np.inf)]
    for band, lo, hi in speed_bands:
        speeds = []
        for player in summary.index:
            column = team_prefix + '_' + player + '_speed'
            player_distance = tracking_data.loc[(tracking_data[column] >= lo) & (tracking_data[column] < hi), column].sum()/25./1000
            speeds.append(player_distance)
        summary[band] = speeds

    nsprints = []
    sprint_threshold = 7
    sprint_window = 1*25
    for player in summary.index:
        column = team_prefix + '_' + player + '_speed'
        player_sprints = np.diff(1*(np.convolve(1*(tracking_data[column]>=sprint_threshold), np.ones(sprint_window), mode='same') >= sprint_window))
        nsprints.append(np.sum(player_sprints == 1))
    summary['# sprints'] = nsprints
    return summary


H_summary = create_summary(H_tracking, 'Home')
A_summary = create_summary(A_tracking, 'Away')


