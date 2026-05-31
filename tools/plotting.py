"""
This module provides functions for displaying images using matplotlib.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from .fileio import ensure_dir
from .colors import (color_palette, colors2plotly)
from pathlib import Path

# Useful to display pretty looking elements in Jupyter notebooks
from IPython.core.display import HTML


# Default color palette of this package.
PALETTE = color_palette("default", alpha=[0.8, 0.45, 0.15], 
                        mix_color="white")
PALETTE_CMAP_CONT_BRG = color_palette(PALETTE,
                                      mix_color="white", as_cmap=True)
PALETTE_CMAP_CONT_RBG = color_palette([PALETTE[1],PALETTE[0], PALETTE[2]],
                                      mix_color="white", as_cmap=True)
PALETTE_CMAP_CONT_BR = color_palette(PALETTE[:2],
                                     mix_color="black", as_cmap=True)
PALETTE_CMAP_CONT_RG = color_palette(PALETTE[1:3],
                                     mix_color="black", as_cmap=True)

PALETTE_CMAP = color_palette("default",
                             mix_color="white", as_cmap=True, n_cmap=3)
PALETTE_PLOTLY = colors2plotly(PALETTE)
PALETTE_RGB = [PALETTE[1], PALETTE[2], PALETTE[0]]


def print_title(title, level=1, sep=None):
    if sep is None:
        sep = "#" if level == 1 else "-"
        
    if level == 1:
        width = max(60, len(title))
    else:
        width = len(title)
    
    if level == 1:
        print()
        print(sep * width)
        print(title)
        print(sep * width)
    else:
        print()
        print(title)
        print(sep * width)

def show_separator():
    """Displays a separator line."""
    display(HTML("<hr>"))
    

def show_header(title=None, 
                subtitle=None, 
                width=9,  # str or (float in inches)
                
                # Default title style
                fontsize=24, 
                color=None,
                align="left",
                
                # Default subtitle style
                subtitle_fontsize=16, 
                subtitle_color=None,
                subtitle_align="left",
                subtitle_kwargs={},

                # Line style
                line_placment=None,
                line_style={},

                **kwargs
                ):
    """Displays a header with a title and a subtitle.

    Args:
        - title: The title of the header.
        - subtitle: The subtitle of the header.
        - width: The width of the header.
        - line_placement: The placement of the line. Can be
                "before_title",
                "after_title",
                "after_subtitle",
                or a list of line placements,
                if None, no line is shown.
    """

    def _display(html):
        display(HTML(html))
        #print(html)

    def add_line(line_style, **kwargs):
        line_style = dict(line_style)
        line_style.setdefault("width", width_str)
        line_style.setdefault("border-top", "2px solid #333333")
        #line_style.setdefault("border-bottom", "none")
        line_style.setdefault("margin", "0 0 0 0")
        line_style.update(kwargs)
        html = """
                <div style="%s"></div>
            """
        html = html % (style2str(line_style))
        return html

    def add_blank_line(height=10):
        html = """
                <div style="height: %dpx;"></div>
            """ % height
        return html
        
    if width is None:
        width_str = "9in"
    elif isinstance(width, str):
        width_str = width
    elif isinstance(width, (int, float)):
        width_str = "%din" % width
    else:
        raise ValueError("Invalid type for width: %s" % type(width))
    
    if line_placment is None:
        line_placment = []
    elif isinstance(line_placment, str):
        line_placment = [line_placment]

    title_style = dict()
    title_style.setdefault("width", width_str)
    title_style.setdefault("text-align", align)
    title_style.setdefault("font-size", "%dpx" % fontsize)
    title_style.setdefault("font-weight", "bold")
    title_style.setdefault("color", color or "#333333")
    title_style.setdefault("border", "none")
    title_style.setdefault("margin", "0px 0 0px 0")
    title_style.setdefault("padding", "10px 0")
    title_style.setdefault("vertical-align", "middle")
    title_style.setdefault("display", "block")
    title_style.setdefault("line-height", "1")
    title_style.update(kwargs)

    # title_style.setdefault("padding", "0px")
    # title_style.setdefault("background-color", "transparent")
    
    subtitle_style = dict()
    subtitle_style.setdefault("width", width_str)
    subtitle_style.setdefault("text-align", subtitle_align)
    subtitle_style.setdefault("font-size", "%dpx" % subtitle_fontsize)
    #subtitle_style.setdefault("font-weight", "bold")
    subtitle_style.setdefault("color", subtitle_color or "#999999")
    subtitle_style.setdefault("border", "none")
    subtitle_style.setdefault("margin", "0px 0 0px 0")
    subtitle_style.setdefault("padding", "10px 0")

    # subtitle_style.setdefault("background-color", "transparent")
    subtitle_style.update(subtitle_kwargs)
    
    # Convert styles to string
    def style2str(style):
        return "; ".join(["%s:%s" % (key, value) for key, value in style.items() if value is not None])  
    
    body = ""

    body += add_blank_line(height=30)

    if "before_title" in line_placment:
        body += add_line(line_style, margin="0 0 0 0")

    if title is not None:
        html = """
                <input
                type="text"
                style="%s"
                value="%s"
                />
            """
        html = html % (style2str(title_style), title)
        body += html

    if "after_title" in line_placment:
        body += add_line(line_style)

    if subtitle is not None:
        html = """
                <input
                type="text"
                style="%s"
                value="%s"
                />
            """
        html = html % (style2str(subtitle_style), subtitle)
        body += html

    if "after_subtitle" in line_placment:
        body += add_line(line_style)


    display(HTML(body))
    #print(body)



# Set default color palett
def setup_plotting(palette=None, no_seaborn=False, **kwargs):
    """
    Adjusts the default plotting settings:
    - Sets the color palette
    """
    if not no_seaborn:
        import seaborn as sns
        sns.set_style("whitegrid")
    
    import matplotlib.pyplot as plt
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)
    plt.rcParams["axes.titleweight"] = 'bold'
    plt.rcParams["grid.linestyle"] = '-'
    plt.rcParams["grid.alpha"] = 0.4
    
    
    #plt.rcParams["figure.figsize"] = (5, 3)
    #plt.rcParams["figure.dpi"] = 300
    for key, value in kwargs.items():
        plt.rcParams[key] = value
        
    # For selectable text in PDFs
    matplotlib.rc("pdf", fonttype=42)
    

def display_image(image=None, scale=None):
    """Displays an image using IPython capabilities."""
    from IPython.display import display
    import PIL.Image
    if isinstance(image, (str, Path)):
        image = PIL.Image.open(image)
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    if scale is not None:
        image = image.resize((int(image.width * scale), 
                              int(image.height * scale)))
    display(image)
    

def draw_image(image, **kwargs):
    """Alias for show_image()"""
    return show_image(image, **kwargs)


def show_image(image,
               ax=None, shape=None, 
               normalize=False,
               normalize_stretch=None,
               scale=None,
               dpi=100, 
               title=None,
               title_kwargs={},
               suppress_info=False,
               background_color=(0.93, 0.93, 0.93),
               show_frame=False,
               frame="deprecated",  # Replaced with show_frame
               frame_color=(0.8,)*4,
               frame_width=2,
               axes_frame=True,
               show_axes=False,
               show_canvas_frame=True,
               box_aspect=None,
               figsize=None,
               save_kwargs={},
               ):
    """Displays an image using matplotlib capabilities.

    Args:
        image: The image to display (as a numpy array or a file path).
        ax:    The image axis to use. If None, a new figure is created.
        shape: The shape of the canvas. If None, the shape is inferred 
               from the image.
        normalize: If True, grayscale images are normalized so that the 
                minimal and maximal values are mapped to black and white, 
                respectively. If normalize is a 2-tuple, the provided values
                will be used for normalization (vmin, vmax = normalize). If 
                False or None, the images are displayed using the full range 
                of the current data type. If "stretch", the contrast is
                stretched using the 1% and 99% percentiles of the intensity.
        normalize_stretch: If not None, the contrast is stretched using the
                provided percentiles of the intensity. Sets normalize to True.
        title: The title of the image.
        title_kwargs: Additional keyword arguments passed to ax.set_title().
        show_frame: If True, a frame is drawn around the image (if the image
               is smaller than the canvas).
        show_canvas_frame: Show boundary of the canvas. Default: True.
        save_kwargs: If not None, a dictionary of keyword arguments passed
                to save_figure() to save the figure.
    """
    if frame!="deprecated":
        show_frame = frame
        print("Warning: The 'frame' argument is deprecated. Use 'show_frame' instead.")

    if isinstance(image, (str, Path)):
        image = plt.imread(image)

    if scale is not None:
        from PIL import Image
        image = Image.fromarray(image)
        image = image.resize((int(image.width * scale),
                              int(image.height * scale)))
        image = np.asarray(image)

    height, width = image.shape[:2]
    
    if ax is None:
        # Create a figure of the right size with 
        # one axis that takes up the full figure
        if figsize is None:
            figsize = width / float(dpi), height / float(dpi)
            if figsize[0]>9 and True:
                figsize = (9, figsize[1] * 9 / figsize[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

    # If the image is grayscale, use the gray colormap.
    cmap = "gray" if len(image.shape) == 2 else None

    vmin, vmax = None, None
    if normalize_stretch is not None:
        if isinstance(normalize_stretch, (float, int)):
            normalize_stretch = (normalize_stretch, 100-normalize_stretch)
        vmin, vmax = np.percentile(image, normalize_stretch)
    elif not normalize:
        # imshow normalization is on by default!
        # If one wants to disable it, one must set vmin and vmax.
        dtype = image.dtype
        if dtype == np.uint8:
            vmin, vmax = 0, 255
        elif dtype == np.uint16:
            vmin, vmax = 0, 65535
        elif dtype in (np.float32, np.float64, float, bool):
            vmin, vmax = 0, 1.0
        else:
            assert False, "Unsupported data type: %s" % dtype
    elif isinstance(normalize, (tuple, list)) and len(normalize) == 2:
        vmin, vmax = normalize
            
    ax.imshow(image, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)

    title = "" if title is None else title
    if not suppress_info:
        title += "\n" if title else ""
        title += "(%s, %s)" % ("x".join(map(str, image.shape)), image.dtype)
    if title:
        ax.set_title(title, **title_kwargs)
    
    # Use fixed axis limits so that
    # the images are shown at scale.
    if shape is not None:
        ax.set_xlim([0, shape[1]])
        ax.set_ylim([shape[0], 0])

    # Use a background color to better see that the images are transparent.
    if background_color is None and (not show_axes):
        ax.axis("off")
    ax.set_facecolor(background_color)
    if not show_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.set_frame_on(axes_frame) 
    ax.set_anchor("N")
    if box_aspect:
        ax.set_box_aspect(box_aspect)
        # Strange hack required if box_aspect is set, but only sometimes.
        ax.update_datalim([[ax.get_xlim()[1], ax.get_ylim()[1]]])

    if not show_canvas_frame:
        for spine in ax.spines.values():
            spine.set_color(None)

    h, w = image.shape[:2]

    if show_frame:
        rect = patches.Rectangle((0, 0), w, h, 
                                linewidth=frame_width, 
                                edgecolor=frame_color, 
                                facecolor='none')

        # Add the rectangle patch to the plot
        ax.add_patch(rect)  
        

    if save_kwargs:
        save_figure(fig=fig, **save_kwargs)
        

def show_image_pair(image1, image2, 
                    title1=None, title2=None, 
                    normalize=True, 
                    dpi=None,
                    figsize=(6, 5),
                    shape="largest",
                    box_aspect=None,
                    frame="deprecated",  # Replaced with show_frame
                    show_frame=True,
                    save_kwargs={},
                    **kwargs):
    """Displays a pair of images side-by-side.

    Args:
        image1: The first image.
        image2: The second image.
        title1: The title of the first image.
        title2: The title of the second image.
        normalize: If True, grayscale images are normalized.
        dpi:    The DPI of the figure.
        show_frame:  If True, a frame is drawn around the images
                (if the images are smaller than the canvas).
                Set "forced" to force a frame.
        save_kwargs: If not None, a dictionary of keyword arguments passed
                to save_figure() to save the figure.
        kwargs:  Additional keyword arguments passed to show_image().
    """
    if frame!="deprecated":
        show_frame = frame
        print("Warning: The 'frame' argument is deprecated. Use 'show_frame' instead.")

    # This converts PIL images to numpy arrays.
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    max_shape = (max(image1.shape[0], image2.shape[0]),
                 max(image1.shape[1], image2.shape[1]))
    
    if dpi is not None:
        # Overrides figsize
        figsize = (max_shape[0]/dpi, max_shape[1]/dpi)

    if shape=="largest":
        shape = max_shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    draw_frame1 = ((show_frame=="forced") or
                   (show_frame and shape is not None and shape!=image1.shape[:2]))
    draw_frame2 = ((show_frame=="forced") or
                   (show_frame and shape is not None and shape!=image2.shape[:2]))
    
    show_image(image1, ax=ax1,
               normalize=normalize, 
               shape=shape, 
               title=title1,
               box_aspect=box_aspect,
               show_frame=draw_frame1,
               **kwargs)
    show_image(image2, ax=ax2,
               normalize=normalize, 
               shape=shape, 
               title=title2,
               box_aspect=box_aspect,
               show_frame=draw_frame2,
               **kwargs) 
    fig.tight_layout()
    
    if save_kwargs:
        save_figure(fig=fig, **save_kwargs)
    
    plt.show()


def show_image_chain(images, **kwargs):
    """Displays a list of images. Equivalent to show_image_grid(..., ncols=-1).
    """
    kwargs.setdefault("ncols", -1)
    show_image_grid(images, **kwargs)


def show_image_grid(images, titles=None, 
                    ncols=3, 
                    scale=4.0, 
                    figsize=None, 
                    shape="largest",
                    dpi=100,
                    suppress_info=False,
                    normalize=True,
                    box_aspect=None,
                    show_frame=True,
                    frame="deprecated",  # Replaced with show_frame
                    header=None,
                    header_kwargs={},
                    save_kwargs={},
                    **kwargs):
    """Displays a grid of images. The width of the grid is determined by ncols.

    Args:
        images: A list of images, a generator of images, or a dictionary of
                titles and images.
        titles: A list of titles for the images (optional). If images are 
                provided as a dictionary, the titles are inferred from the
                dictionary keys.
        ncols:  The number of columns in the grid. If ncols=-1, the number of 
                columns is set to the number of images.
        scale:  The scale of the figure.
        figsize: The size of the figure. (Overrides scale.)
        shape:  The shape of the image canvas. Default: "largest". If 
                "largest", the shape is set to the largest image, and all 
                images are shown at the same scale. If None, the shape is 
                inferred from the images. 
        dpi:    The DPI of the figure. Default: 100. Only relevant if
                figsize is not set. The real DPI will not be exactly 
                the same as the requested DPI.
        suppress_info: If True, the image shape and data type are not shown
                in the title.
        normalize: If True, grayscale images are normalized so that the 
                minimal and maximal values are mapped to black and white, 
                respectively. If normalize is a 2-tuple, the provided values
                will be used for normalization (vmin, vmax = normalize). If 
                False or None, the images are displayed using the full range 
                of the current data type. 
        box_aspect: If not None, the aspect ratio of the image is fixed.
                Useful for images with different aspect ratios.
        show_frame: If True, a frame is drawn around the images (if the images
                are smaller than the canvas). Set "forced" to force a frame.
        header: If not None, a header is displayed above the images.
        header_kwargs: Additional keyword arguments passed to show_header().
        save_kwargs: If not None, a dictionary of keyword arguments passed
                to save_figure() to save the figure.
        kwargs:  Additional keyword arguments passed to show_image().
        
    Usage:
        show_image_grid([image1, image2, image3]) 
        show_image_grid({title1: image1, title2: image2, title3: image3})
    """
    if frame!="deprecated":
        show_frame = frame
        print("Warning: The 'frame' argument is deprecated. Use 'show_frame' instead.")

    if not images:
        return
    
    # Manage input types
    import types
    if isinstance(images, types.GeneratorType):
        images = list(images)
    elif isinstance(images, dict):
        titles = list(images.keys())
        images = list(images.values())
    if isinstance(titles, types.GeneratorType):
        titles = list(titles)
    elif titles is None:
        titles = [None] * len(images)
    
    images = [None if img is None else np.asarray(img) for img in images]
    assert titles is None or (len(images) == len(titles))
    has_titles = any(titles)

    # Number of rows and columns
    if ncols == -1:
        ncols = len(images)
        nrows = 1
    else:
        ncols = min(len(images), ncols)
        nrows = int(np.ceil(len(images) / ncols))

    # Manage shape
    h_max, w_max = np.vstack([ img.shape[:2] for img in images if img is not None ]).max(axis=0)
    
    # Height of title in inches
    h_title = (has_titles*1 + (not suppress_info)*0.5)*scale
    h_fig = (h_max / dpi * scale + h_title)*nrows
    w_fig = (w_max / dpi * scale)*ncols

    if shape == "largest":
        shape = (h_max, w_max)

    # Figure size
    if figsize is None:
        figsize = (w_fig, h_fig)
    # Limit the figure size to 9 inches in width.
    # this is a good size for printing
    if figsize[0]>9:
        figsize = (9, figsize[1] * 9 / figsize[0])
        
    if header:
        show_header(title=header,
                    width=figsize[0],
                    fontsize=24,
                    kwargs=header_kwargs)

    # Create the figure
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=figsize,
                             squeeze=False)
    
    for ax, image, title in zip(axes.flat, images, titles):    
        if image is None:
            ax.axis("off")
            continue        
        draw_frame = ((show_frame=="forced") or
                      (show_frame and shape is not None and shape!=image.shape[:2]))
        show_image(image, title=title, ax=ax, 
                   suppress_info=suppress_info, 
                   normalize=normalize,
                   box_aspect=box_aspect,
                   shape=shape,
                   show_frame=draw_frame,
                   **kwargs)
        
    # Disable grid axes that are not used
    for i in range(len(images), len(axes.flat)):
        axes.flat[i].axis("off")
    fig.tight_layout()
    
    if save_kwargs:
        save_figure(fig=fig, **save_kwargs)
    
    plt.show()


def save_figure(fig=None, path="figure.pdf",
                subdir_ext=False,
                **kwargs):
    if fig is None:
        fig = plt.gcf()
    kwargs.setdefault("transparent", True)
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("dpi", 300)
    path = Path(path)
    if subdir_ext:
        ext = path.suffix.lstrip(".").lower()
        subdir = path.parent / ext
        path = subdir / path.name

    ensure_dir(path.parent)
    plt.savefig(path, **kwargs)
