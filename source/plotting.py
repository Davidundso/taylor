import matplotlib.pyplot as plt
import numpy as np
import tueplots.bundles as bundles
import tueplots.fontsizes as fontsizes



def get_style(font_adjustment=0, rel_width=1):
    # Retrieve the original style from jmlr2001
    jmlr_style = bundles.jmlr2001(rel_width=rel_width)

    # Update the style with the desired font sizes, applying the font_adjustment
    jmlr_style.update({
        'font.size': 10.95 + font_adjustment,
        'axes.labelsize': 10.95 + font_adjustment,
        'legend.fontsize': 8.95 + font_adjustment,
        'xtick.labelsize': 8.95 + font_adjustment,
        'ytick.labelsize': 8.95 + font_adjustment,
        'axes.titlesize': 10.95 + font_adjustment,
    })

    return jmlr_style





# Define the general plot function
def plot_data(data, colors, size: str, x_axis_title: str, y_axis_title: str, use_legend = False, legend_labels=None, concat_data=False, horizontal=True, 
              font_adj=1):
    """
    General function to plot data with thesis-compliant style using jmlr2001 format.
    
    Arguments:
    - data: list of data to be plotted, each data entry is for one line/plot.
    - colors: list of colors for each line.
    - size: "large" or "small", adjusts plot size accordingly by modifying rel_width.
    - x_axis_title: string, title for the x-axis.
    - y_axis_title: string, title for the y-axis.
    - use_legend: boolean, whether to use a legend.
    - legend_labels: list of labels for the legend (optional).
    - concat_data: boolean, if True concatenates data along the x-axis.
    - font_adj: Float that adjusts the font w
    """
    default_colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan']
    # Set rel_width based on the size argument
    if size == "large":
        rel_width = 1.0  # Default full size
    elif size == "small":
        rel_width = 0.5  # Half width for small plots
    else:
        raise ValueError("Size must be 'large' or 'small'.")

    

    # Konfiguration für JMLR 2001-Stil mit angepasster Schriftgröße
    jmlr_style = get_style(font_adjustment=font_adj, rel_width=rel_width)

    
    # Use the jmlr style for the plot
    with plt.rc_context(jmlr_style):
        plt.figure(dpi=300)

        # Initialize start index if we are concatenating along the x-axis
        start_index = 0

        # Plot each dataset
        for idx, dataset in enumerate(data):
            data_np = np.array([p.detach().numpy() for p in dataset])  # Detach if they're tensors

            # If concat_data is True, update x-values to concatenate the data along the x-axis
            if concat_data:
                x_values = np.arange(start_index, start_index + len(data_np))
                start_index += len(data_np)  # Update starting index for next dataset
            else:
                x_values = np.arange(len(data_np))  # Regular plotting (no concatenation)

            # Use specified color or default color from the color cycle
            if colors and idx < len(colors):
                color = colors[idx]
            else:
                color = default_colors[idx % len(default_colors)]  # Use default color cycle
            
            label = legend_labels[idx] if legend_labels and idx < len(legend_labels) else None
            
            # Plot the line
            plt.plot(x_values, data_np, color=color, label=label, alpha=0.6)

        # Set the x and y axis titles
        plt.xlabel(x_axis_title)
        plt.ylabel(y_axis_title)
        
        # Add legend if required
        if use_legend and legend_labels:
            plt.legend()

        if horizontal:
        # Add grid and horizontal line at y=0
            plt.axhline(0, color='black', lw=0.8, ls='--')
        
        plt.grid(True)

        # Show and save the plot
        plt.savefig('plot.pdf')
        plt.show()




def test():
    print("import succesful :)")
# dictionary kopieren + umbenennen
# dictionary überschreiben
# aus anderen files importieren: from source.plotting import <<name dictionary>>
# über rel width (als argument) größe des Plots anpassen



