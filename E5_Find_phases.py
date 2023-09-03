"""
This file isolates the phases that will be analysed
by producing dataframes with start and end_frame values
These include, Build-Ups, Established Possession, 
Goal Kicks, Chances created
"""




# Function to calculate mean positions
def mean_position(team_tracking):
    mean_tracking = pd.DataFrame(team_tracking.mean()).transpose()
    mean_tracking.index = ['Mean']
    return mean_tracking

# Label the deepest players from each team
def label_defenders(team_means, H_or_A):
    # Get the columns corresponding to '_x' but not 'ball'
    x_columns = [col for col in team_means.columns if '_x' in col and 'ball' not in col]
    # Get the mean values for these columns
    x_means = pd.to_numeric(team_means.loc['Mean', x_columns], errors='coerce')
    # Determine whether to use nsmallest or nlargest based on the argument passed to the function
    if H_or_A == "home":
        key_players = set(x_means.nlargest(7).index)
    elif H_or_A == "away":
        key_players = set(x_means.nsmallest(7).index)
    else:
        raise ValueError("'H_or' must be either 'home' or 'away'")
    # Add a row to mark which players are defenders
    team_means.loc['Defenders', :] = 0
    team_means.loc['Defenders', key_players] = 1
    return team_means

# Idenitfy passes between the deepest players from each team
def identify_defender_passing(team_events, team_means, H_or_A):
    # Get the defender player IDs
    defender_ids = [int(col.split('_')[1]) for col in team_means.columns if team_means.loc['Defenders', col] == 1]

    # Initialize the new column with zeros
    team_events['DefenderPass'] = 0

    # Set threshold based on H_or
    if H_or_A == 'home':
        threshold = 30
    elif H_or_A == 'away':
        threshold = -30
    else:
        raise ValueError("'H_or' must be either 'home' or 'away'")

    # Iterate over the rows of the DataFrame
    for i, row in team_events.iterrows():
        # Check if the event is a pass
        if row['Type'] == 'PASS':
            # Extract numeric player IDs from the 'From' and 'To' columns
            from_id = int(''.join(filter(str.isdigit, row['From'])))
            to_id = int(''.join(filter(str.isdigit, row['To'])))

            # Check if the pass was made by a defender
            if from_id in defender_ids:
                # Check if the pass was received by a defender
                if to_id in defender_ids:
                    # Check if the pass was made in the deepest third of the pitch
                    if (H_or_A == 'home' and row['Start X'] > threshold) or \
                       (H_or_A == 'away' and row['Start X'] < threshold):
                        # If all conditions are met, set DefenderPass to 1
                        team_events.at[i, 'DefenderPass'] = 1
    return team_events

# Pick out sequnces that begin with defender passing to each other
# And sequences that end with losing the ball or going over halfway
def create_build_sequences(team_events, team_tracking, H_or_A):
    # Initialize list for start and end frames
    start_frames = []
    end_frames = []

    # Initialize a flag for active sequence
    active_sequence = False

    # Iterate over the rows of DataFrame
    for i, row in team_events.iterrows():
        # Check if the event is a 'DefenderPass'
        if row['DefenderPass'] == 1:
            if active_sequence:
                # Ignore this row if another sequence is already active
                continue
            # If not, start a new sequence
            start_frames.append(row['Start Frame'])
            active_sequence = True
        # If in active sequence, check for non-'PASS' event or if the ball enters 10 meters into the opposing half
        elif active_sequence:
            ball_x_position = team_tracking.loc[team_tracking.index == row['End Frame'], 'ball_x'].item()
            if row['Type'] != 'PASS' or (H_or_A == 'home' and ball_x_position > 10) or (H_or_A == 'away' and ball_x_position < -10):
                end_frames.append(row['End Frame'] + 3*25)
                active_sequence = False

    # If last sequence was not ended, end it at the last frame
    if len(start_frames) > len(end_frames):
        end_frames.append(team_events['End Frame'].iloc[-1] + 3*25)

    # Create DataFrame from start and end frames
    sequences = pd.DataFrame({'start_frame': start_frames, 'end_frame': end_frames})
    return sequences





# Locate the frame of GK pass, and create a sequnce of frames around it.
def locate_gk_pass_frames(team_events, keeper):
    # Filter events where Type is PASS and From contains keeper
    pass_events = team_events[(team_events['Type'] == 'PASS') & (team_events['From'].str.contains(keeper))]
    # Rename Frame column to pass_frame
    pass_events = pass_events.rename(columns={'Start Frame': 'pass_frame'})

    # Calculate start_frame and end_frame
    pass_events['start_frame'] = pass_events['pass_frame'] - (2*25)
    pass_events['end_frame'] = pass_events['pass_frame'] + (8*25)

    # Select necessary columns
    output = pass_events[['start_frame', 'pass_frame', 'end_frame']]
    return output





# Locate the frames of passes before shots, create sequnce around it.
def locate_pre_shot_pass_frames(team_events):
    # Find events where the Type is 'SHOT'
    team_events = team_events.reset_index(drop=True)
    shot_events = team_events[team_events['Type'] == 'SHOT']

    # Initialize the output DataFrame
    output = pd.DataFrame(columns=['start_frame', 'pass_frame', 'end_frame'])

    # Iterate over the shot events and locate the preceding pass
    for _, row in shot_events.iterrows():
        # Find the event directly preceding the shot event
        preceding_event = team_events.loc[row.name - 1]

        # Check if the preceding event is a pass
        if preceding_event['Type'] == 'PASS':
            pass_frame = preceding_event['Start Frame']
            start_frame = pass_frame - (2 * 25)
            end_frame = pass_frame + (8 * 25)
            output.loc[len(output)] = [start_frame, pass_frame, end_frame]
    return output





# Identify passes in possession phase, near to or in opposing half.
def identify_possession_phase(team_events, H_or_A):
    if H_or_A == 'home':
        indices = team_events[(team_events['Type'] == 'PASS') & 
                            (team_events['End X'] <= 10)].index
    elif H_or_A == 'away':
        indices = team_events[(team_events['Type'] == 'PASS') & 
                            (team_events['End X'] >= -10)].index
    # Initialize the new column with zeros
    team_events.loc[:, 'PossessionPhase'] = 0
    # Set 'PossessionPhase' to 1 for the selected events
    team_events.loc[indices, 'PossessionPhase'] = 1
    return team_events

# Create sequences based on these possession passes, it starts with one
# of these passes, ends with a return to build-up, losing the ball, or a shot
# sequence must be a minimum of three passes long
def create_possession_sequences(team_events):
    start_frames = []
    end_frames = []
    i = 0
    while i < len(team_events):
        # Check if this event has 'PossessionPhase' == 1
        if team_events.iloc[i]['PossessionPhase'] == 1:
            start = i
            pass_count = 0  # Initialize counter for 'PASS' events
            # Look through rows until a row has a 'Type' that isn't 'PASS' or 'SHOT'
            # Or there are 3 passes in a row which do not have 'PossessionPhase' == 1
            while i < len(team_events) and (team_events.iloc[i]['Type'] in ['PASS', 'SHOT'] or 
                                             (i < len(team_events)-2 and 
                                              team_events.iloc[i+1]['PossessionPhase'] == 1 and 
                                              team_events.iloc[i+2]['PossessionPhase'] == 1 and
                                              team_events.iloc[i+3]['PossessionPhase'] == 1)):
                if team_events.iloc[i]['Type'] == 'PASS':
                    pass_count += 1
                i += 1
            
            # Only add the sequence if it contains at least 3 'PASS' events and
            # the difference between the end frame and the start frame is at least 250 (10 seconds)
            if pass_count >= 3 and (team_events.iloc[i]['End Frame'] - team_events.iloc[start]['Start Frame'] >= 10*25):
                start_frames.append(team_events.iloc[start]['Start Frame'])
                end_frames.append(team_events.iloc[i]['End Frame'])
        i += 1

    sequences = pd.DataFrame({'start_frame': start_frames, 'end_frame': end_frames})
    return sequences



"""
The functions are then executed to produce the dataframes
of sequences that can be analyse later on.
"""

# tracking = remove_sub_columns(tracking)
# tracking = remove_sub_columns(tracking)
# tracking = remove_postsub_rows(tracking)
# tracking = remove_postsub_rows(tracking)



# Get mean player positions
H_team_XY_means = mean_position(H_tracking)
A_team_XY_means = mean_position(A_tracking)
# Label deepest players
H_team_XY_means = label_defenders(H_team_XY_means, 'home')
A_team_XY_means = label_defenders(A_team_XY_means, 'away')
# Add labels to events
H_events = identify_defender_passing(H_events, H_team_XY_means, "home")
A_events = identify_defender_passing(A_events, A_team_XY_means, "away")
# Reset the index of H_events for continuous indexing
H_events.reset_index(drop=True, inplace=True)
A_events.reset_index(drop=True, inplace=True)
# Get the indexes where DefenderPass is 1
H_build_up_start_passes = H_events[H_events['DefenderPass'] == 1].index.tolist()
H_build_up_start_passes = A_events[A_events['DefenderPass'] == 1].index.tolist()
# Get build-ups sequences of each team
H_build_up_sequences = create_build_sequences(H_events, H_tracking, 'home')
A_build_up_sequences = create_build_sequences(A_events, A_tracking, 'away')



# Use Metrica functions to find goalkeeper
H_keeper_number = mio.find_goalkeeper(H_tracking)
A_keeper_number = mio.find_goalkeeper(A_tracking)
# Locate the frames where the GK passes the ball
H_gk_pass_sequences = locate_gk_pass_frames(H_events, str(H_keeper_number))
A_gk_pass_sequences = locate_gk_pass_frames(A_events, str(A_keeper_number))
# Filter GK's passes from H_events and create sequences
H_gk_pass_events = H_events[H_events['Start Frame'].isin(H_gk_pass_sequences['pass_frame'])]
A_gk_pass_events = A_events[A_events['Start Frame'].isin(A_gk_pass_sequences['pass_frame'])]



# Locate the frames where passes precede shots
H_pre_shot_pass_sequences = locate_pre_shot_pass_frames(H_events)
A_pre_shot_pass_sequences = locate_pre_shot_pass_frames(A_events)
# Filter the passes that precede shots from H_events and A_events, create sequences
H_pre_shot_pass_events = H_events[H_events['Start Frame'].isin(H_pre_shot_pass_sequences['pass_frame'])]
A_pre_shot_pass_events = A_events[A_events['Start Frame'].isin(A_pre_shot_pass_sequences['pass_frame'])]



# Identify where Possession pass takes place
identify_possession_phase(H_events, 'home')
identify_possession_phase(A_events, 'away')
# Create sequences for poessession periods
H_possess_sequences = create_possession_sequences(H_events)
A_possess_sequences = create_possession_sequences(A_events)

