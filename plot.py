
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


# function to plot the state-action value........................................................
def plot_action_value(mdp, action_value_grid, plot_policy = False, policy = None):
    top=action_value_grid[:,0].reshape((5,5))
    top_value_positions = [(0.38,0.25),(1.38,0.25),(2.38,0.25),(3.38,0.25),(4.38,0.25),
                           (0.38,1.25),(1.38,1.25),(2.38,1.25),(3.38,1.25),(4.38,1.25),
                           (0.38,2.25),(1.38,2.25),(2.38,2.25),(3.38,2.25),(4.38,2.25),
                           (0.38,3.25),(1.38,3.25),(2.38,3.25),(3.38,3.25),(4.38,3.25),
                           (0.38,4.25),(1.38,4.25),(2.38,4.25),(3.38,4.25),(4.38,4.25)]
    
    bottom=action_value_grid[:,1].reshape((5,5))
    bottom_value_positions = [(0.38,0.8),(1.38,0.8),(2.38,0.8),(3.38,0.8),(4.38,0.8),
                           (0.38,1.8),(1.38,1.8),(2.38,1.8),(3.38,1.8),(4.38,1.8),
                           (0.38,2.8),(1.38,2.8),(2.38,2.8),(3.38,2.8),(4.38,2.8),
                           (0.38,3.8),(1.38,3.8),(2.38,3.8),(3.38,3.8),(4.38,3.8),
                           (0.38,4.8),(1.38,4.8),(2.38,4.8),(3.38,4.8),(4.38,4.8)]
    
    left=action_value_grid[:,2].reshape((5,5))
    left_value_positions = [(0.05,0.5),(1.05,0.5),(2.05,0.5),(3.05,0.5),(4.05,0.5),
                           (0.05,1.5),(1.05,1.5),(2.05,1.5),(3.05,1.5),(4.05,1.5),
                           (0.05,2.5),(1.05,2.5),(2.05,2.5),(3.05,2.5),(4.05,2.5),
                           (0.05,3.5),(1.05,3.5),(2.05,3.5),(3.05,3.5),(4.05,3.5),
                           (0.05,4.5),(1.05,4.5),(2.05,4.5),(3.05,4.5),(4.05,4.5)]
    
    right=action_value_grid[:,3].reshape((5,5))
    right_value_positions = [(0.65,0.5),(1.65,0.5),(2.65,0.5),(3.65,0.5),(4.65,0.5),
                           (0.65,1.5),(1.65,1.5),(2.65,1.5),(3.65,1.5),(4.65,1.5),
                           (0.65,2.5),(1.65,2.5),(2.65,2.5),(3.65,2.5),(4.65,2.5),
                           (0.65,3.5),(1.65,3.5),(2.65,3.5),(3.65,3.5),(4.65,3.5),
                           (0.65,4.5),(1.65,4.5),(2.65,4.5),(3.65,4.5),(4.65,4.5)]
    
    
    fig, ax=plt.subplots(figsize=(12,7))
    ax.set_ylim(5, 0)
    tripcolor = quatromatrix(left, top, right, bottom, ax=ax,
                 triplotkw={"color":"k", "lw":1},
                 tripcolorkw={"cmap": "coolwarm"}) 

    ax.margins(0)
    ax.set_aspect("equal")
    fig.colorbar(tripcolor)

    for i, (xi,yi) in enumerate(top_value_positions):
        plt.text(xi,yi,round(top.flatten()[i],1), size=11, color="w")
    for i, (xi,yi) in enumerate(right_value_positions):
        plt.text(xi,yi,round(right.flatten()[i],1), size=11, color="w")
    for i, (xi,yi) in enumerate(left_value_positions):
        plt.text(xi,yi,round(left.flatten()[i],1), size=11, color="w")
    for i, (xi,yi) in enumerate(bottom_value_positions):
        plt.text(xi,yi,round(bottom.flatten()[i],1), size=11, color="w")

    if plot_policy:
        action_to_arrow = {
        'up': (0, 0.5),
        'down': (0, -0.5),
        'left': (-0.5, 0),
        'right': (0.5, 0)
        }


    # plot the policy..............................................................................
        for y in range(mdp.gridsize[0]):
            for x in range(mdp.gridsize[1]):
                actions = policy[y, x]
                for action in actions:
                    dx, dy = action_to_arrow[action]
                    plt.arrow(x + 0.5, y + 0.6, dx*0.3, -dy*0.3, head_width=0.2, head_length=0.2, fc='green', ec = 'green')

    # plt.show()
    return ax
    


def quatromatrix(left, bottom, right, top, ax=None, triplotkw={},tripcolorkw={}):

    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(), 
              right.flatten(), top.flatten()   ].flatten()

    triplot = ax.triplot(A[:,0], A[:,1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor



# Function to plot the policy ..........................................................................................
def policy_plot(mdp, policy):
    # Define the grid size
    rows, cols = mdp.gridsize[0], mdp.gridsize[1]

    # Generate random values for each section
    # Each cell will have 4 values: right, up, left, bottom
    # values = np.random.randint(1, 10, (rows, cols, 4))

    # Create the plot
    fig, ax = plt.subplots(figsize = (7,5))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(0, cols+1, 1))
    ax.set_yticks(np.arange(0, rows+1, 1))
    ax.grid(True)

    ax.invert_yaxis()

    # Plot each cell
    for row in range(rows):
        for col in range(cols):

            if not mdp.isTerminalState((row, col)):
                # Get the values for the current cell
                up, bottom, left, right = policy[(row, col)]



                # Create a rectangle for the cell background color
                if row == 0 and col == 1:
                    rect1 = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor= 'b')
                    ax.add_patch(rect1)
                if row == 0 and col == 4:
                    rect2 = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor= 'g')
                    ax.add_patch(rect2)

                if row == 4 and col == 2:
                    rect5 = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor= 'r')
                    ax.add_patch(rect5)
                if row == 4 and col == 4:
                    rect6 = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor= 'y')
                    ax.add_patch(rect6)
                
                # Plot the sections
                ax.text(col + 0.75, row + 0.5, str(round(right,2)), va='center', ha='center', color='black')
                ax.text(col + 0.5,  row + 0.25, str(round(up,2)), va='center', ha='center', color='cyan')
                ax.text(col + 0.25, row + 0.5, str(round(left, 2)), va='center', ha='center', color='gold')
                ax.text(col + 0.5, row + 0.75, str(round(bottom,2)), va='center', ha='center', color='purple')

            else:
                rect = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor= 'black')
                ax.add_patch(rect)

    return ax