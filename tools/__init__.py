__version__ = "0.3"
__author__ = "Norman Juchler"

from .plotting import (print_title,
                       show_header,
                       show_separator,
                       show_image,
                       show_image_pair, 
                       show_image_chain, 
                       show_image_grid,
                       save_figure,
                       setup_plotting,
                       display_image,
                       PALETTE,
                       PALETTE_RGB,
                       PALETTE_CMAP_CONT_BR,
                       PALETTE_CMAP_CONT_RG,
                       PALETTE_CMAP_CONT_BRG,
                       PALETTE_CMAP_CONT_RBG,
                       PALETTE_CMAP,
                       PALETTE_PLOTLY)
from .plotting_ml import (plot_decision_boundary)
from .colors import (color_palette,
                     colors2plotly,
                     color_transition,
                     color_transitions)
from .fileio import (load_audio,
                     ensure_dir)
