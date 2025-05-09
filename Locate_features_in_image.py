import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Try to import tkinter, set a flag if successful
try:
    import tkinter as tk
    from tkinter import filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


def label_locator_app():
    if TKINTER_AVAILABLE:
        # Set up Tkinter root window (will be hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Open file dialog to select image
        image_path = filedialog.askopenfilename(
            title='Select an Image File',
            filetypes=[
                ('Image files', '*.png *.jpg *.jpeg *.bmp *.gif, *.tif, *.tiff'),
                ('All files', '*.*'),
            ],
        )

        # Check if user canceled the dialog
        if not image_path:
            print('No file selected. Exiting.')
            root.destroy()
            return

        image_path = Path(image_path)
    else:
        # Ask user for image path
        image_path = input('Please enter the path to the image file: ')
        image_path = Path(image_path.strip("'").strip())  # Handle quoted paths

    # Check if file exists
    if not image_path.is_file():
        print(f'Error: File "{image_path}" not found.')
        return

    # Load the image
    try:
        img = mpimg.imread(image_path)
    except Exception as e:
        print(f'Error loading image: {e}')
        return

    # Initialize state
    flipped = False
    clicks = []

    # Function to handle mouse clicks
    def on_click(event):
        nonlocal flipped, img
        if event.inaxes == ax:
            if event.button == 1:  # Left-click to select points
                clicks.append((int(event.xdata), int(event.ydata)))
                ax.plot(event.xdata, event.ydata, 'ro')  # Mark click with red dot
                fig.canvas.draw()
                if len(clicks) == 2:
                    plt.close(fig)  # Close after 2 clicks
            elif event.button == 3:  # Right-click to flip
                img = np.fliplr(img)  # Flip left-to-right
                flipped = not flipped
                ax.imshow(img, origin='lower')
                # Redraw existing clicks if any
                if clicks:
                    ax.plot([c[0] for c in clicks], [c[1] for c in clicks], 'ro')
                fig.canvas.draw()

    # Set up the plot
    fig, ax = plt.subplots()
    ax.imshow(img, origin='lower')
    ax.set_title('1. Right-click to flip image (if required)\n2. Left-click 2 label locations')

    # Connect event handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Show the plot
    plt.show()

    # Report results
    if len(clicks) == 2:
        # Construct the features dictionary with clicked coordinates
        features = [
            {
                'label': '<FIRST CLICKED LABEL NAME HERE>',
                'feature_location': [int(clicks[0][0]), int(clicks[0][1])],  # First click
            },
            {
                'label': '<SECOND CLICKED LABEL NAME HERE>',
                'feature_location': [int(clicks[1][0]), int(clicks[1][1])],  # Second click
            },
        ]
        result = {'features': features}

        # Print as formatted JSON for easy copy-paste
        print(
            '\nResults (please copy and paste into config file, adding relevant label names):\n\n'
        )
        print(json.dumps(result, indent=4)[2:-2])

        # Also print flip status separately
        if flipped:
            print('\n\nRemember to flip all images.  Using "flipped.py"')
    else:
        print('Error: Did not receive 2 clicks before closing.')


if __name__ == '__main__':
    label_locator_app()
