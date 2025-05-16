import matplotlib.pyplot as plt
import numpy as np
import tueplots.bundles as bundles
import tueplots.fontsizes as fontsizes



def get_style(font_adjustment=1, rel_width=1):
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

def get_colors():
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
    return default_colors





# Define the general plot function
def plot_data(data, custom_x_axis, colors, size: str, x_axis_title: str, y_axis_title: str, use_legend = False,
               legend_labels=None, concat_data=False, horizontal=True, font_adj=1, show=True, yscale='linear', x_start=0, y_lim=None):
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
    - yscale: string, scale for the y-axis (linear, log, symlog, logit).
    - y_lim: tuple, limits for the y-axis.
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

    if x_start != 0:
        data = [data[x_start - 1:] for data in data]

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

             # Check if custom_x_axis is provided and not empty
            if len(custom_x_axis) > 0 and len(custom_x_axis) == len(data_np):
            # Use custom_x_axis if provided and matches the data length
                x_values = custom_x_axis
            else:
            # If concat_data is True, update x-values to concatenate the data along the x-axis
                if concat_data:
                    x_values = np.arange(start_index, start_index + len(data_np)) + x_start  # Concatenated x-values
                    start_index += len(data_np)  # Update starting index for next dataset
                else:
                    x_values = np.arange(len(data_np)) + x_start  # Regular plotting (no concatenation)

            # Use specified color or default color from the color cycle
            if colors and idx < len(colors):
                color = colors[idx]
            else:
                color = default_colors[idx % len(default_colors)]  # Use default color cycle
            
            label = legend_labels[idx] if legend_labels and idx < len(legend_labels) else None
            
            # Plot the line
            plt.plot(x_values, data_np, color=color, label=label, alpha=0.6)

        # Apply the y-axis scale
        plt.yscale(yscale)
        
        if y_lim is not None:
            plt.ylim(y_lim)
        
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

        plt.xticks(np.append(plt.gca().get_xticks(), x_start))
        plt.xlim(xmin=x_start)

        # Show and save the plot
        plt.savefig('plot.pdf')
        if show:
            plt.show()

# includes standard deviation
def plot_data1(data, custom_x_axis, colors, size: str, x_axis_title: str, y_axis_title: str, use_legend=False,
              legend_labels=None, concat_data=False, horizontal=True, font_adj=1, show=True, yscale='linear',
              x_start=0, y_lim=None, std_data=None, std_legend_label=None, scientific_x_axis=False):
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
    - font_adj: Float that adjusts the font.
    - yscale: string, scale for the y-axis (linear, log, symlog, logit).
    - y_lim: tuple, limits for the y-axis.
    - std_data: list (optional), where each entry corresponds to a dataset in `data` and contains
                the standard deviation (or similar variability measure) for each data point.
                If provided, a grey area representing data ± std is drawn.
    - std_legend_label: string (optional), label for the grey area representing standard deviation.
                If provided, this label is added to the legend (only once).
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

    if x_start != 0:
        data = [data[x_start - 1:] for data in data]

    # Konfiguration für JMLR 2001-Stil mit angepasster Schriftgröße
    jmlr_style = get_style(font_adjustment=font_adj, rel_width=rel_width)

    # Flag to add the legend entry for the grey area only once.
    grey_legend_added = False

    # Use the jmlr style for the plot
    with plt.rc_context(jmlr_style):
        plt.figure(dpi=300)

        # Initialize start index if we are concatenating along the x-axis
        start_index = 0

        # Plot each dataset
        for idx, dataset in enumerate(data):
            data_np = np.array([p.detach().numpy() for p in dataset])  # Detach if they're tensors

            # Check if custom_x_axis is provided and not empty
            if len(custom_x_axis) > 0 and len(custom_x_axis) == len(data_np):
                # Use custom_x_axis if provided and matches the data length
                x_values = custom_x_axis
            else:
                # If concat_data is True, update x-values to concatenate the data along the x-axis
                if concat_data:
                    x_values = np.arange(start_index, start_index + len(data_np)) + x_start  # Concatenated x-values
                    start_index += len(data_np)  # Update starting index for next dataset
                else:
                    x_values = np.arange(len(data_np)) + x_start  # Regular plotting (no concatenation)

            # Use specified color or default color from the color cycle
            if colors and idx < len(colors):
                color = colors[idx]
            else:
                color = default_colors[idx % len(default_colors)]  # Use default color cycle
            
            label = legend_labels[idx] if legend_labels and idx < len(legend_labels) else None
            
            # Plot the line
            plt.plot(x_values, data_np, color=color, label=label, alpha=0.6)

            # Draw grey area if std_data is provided for this dataset.
            if std_data is not None and idx < len(std_data):
                std_vals = np.array(std_data[idx])
                lower_bound = data_np - std_vals
                upper_bound = data_np + std_vals
                if not grey_legend_added and std_legend_label is not None:
                    plt.fill_between(x_values, lower_bound, upper_bound, color='grey', alpha=0.2,
                                     label=std_legend_label)
                    grey_legend_added = True
                else:
                    plt.fill_between(x_values, lower_bound, upper_bound, color='grey', alpha=0.2)

        # Apply the y-axis scale
        plt.yscale(yscale)
        
        if y_lim is not None:
            plt.ylim(y_lim)
        
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

        plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0)) if scientific_x_axis else plt.ticklabel_format(style='plain', axis='x')

        plt.xticks(np.append(plt.gca().get_xticks(), x_start))
        plt.xlim(xmin=x_start)

        # Show and save the plot
        plt.savefig('plot.pdf')
        if show:
            plt.show()



def barplot_data(values, colors, y_axis_title='Values', x_axis_title='Categories',
                  legend_labels=None, size: str = "large", font_adj=1, show=True):
    """
    Function to plot a bar plot with thesis-compliant style using jmlr2001 format.

    Arguments:
    - values: List of values to be plotted as bars.
    - colors: List of colors for each bar.
    - legend_labels: List of labels for the legend (optional).
    - size: "large" or "small", adjusts plot size accordingly by modifying rel_width.
    - font_adj: Float that adjusts the font size.
    - show: Whether to display the plot after it's created.
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

    # Get JMLR style configuration
    jmlr_style = get_style(font_adjustment=font_adj, rel_width=rel_width)

    # Use the jmlr style for the plot
    with plt.rc_context(jmlr_style):
        plt.figure(dpi=300)

        # Number of bars
        num_bars = len(values)
        
        # X positions for bars
        x_pos = np.arange(num_bars)

        # Use specified colors or default color cycle
        if colors:
            bar_colors = colors
        else:
            bar_colors = default_colors[:num_bars]  # Use the first n colors if fewer colors are specified

        # Plot the bars
        plt.bar(x_pos, values, color=bar_colors, alpha=0.6)

        # Set x-ticks and labels
        plt.xticks(x_pos, legend_labels if legend_labels else [f'Bar {i+1}' for i in range(num_bars)])

        # Set the y-axis title
        plt.ylabel(y_axis_title)

        # Set font adjustments for axis labels
        plt.xlabel(x_axis_title)

        # Show and save the plot
        plt.savefig('barplot.pdf')
        if show:
            plt.show()





def test():
    print("import succesful :)")
# dictionary kopieren + umbenennen
# dictionary überschreiben
# aus anderen files importieren: from source.plotting import <<name dictionary>>
# über rel width (als argument) größe des Plots anpassen



