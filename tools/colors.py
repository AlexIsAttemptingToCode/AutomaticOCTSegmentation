import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mplc


def color_palette(name="default",
                  desat=None,
                  alpha=None,
                  mix_color=None,
                  grouped=False,
                  as_cmap=False,
                  n_cmap=256,
                  show=False,
                  sns_kwargs={}):
    """
    Returns a list of colors.

    Arguments:
        name:       Name of the palette.
        desat:      Desaturation factor. 0: full desaturation, 1: no change.
                    If a list, apply desaturation multiple times and combine
                    the resulting list of colors. 
        alpha:      Alpha value. 0: full transparency, 1: no change.
                    If a list, apply alpha multiple times and combine the
                    resulting list of colors.
        mix_color:  Mix the palette with a color. If None, no mixing is done.
                    Can be a string, a 3-tuple or a 4-tuple. Useful to create
                    color sequences that match a certain background color. The
                    alpha value of mix_color is enforced
        grouped:    If True, keep values grouped by desaturation and alpha.
                    Does not have any effect if desat or alpha are not lists.
        as_cmap:    If True, return a colormap (with n_colors=n_cmap).
        n_cmap:     Number of colors in the colormap (default: n_cmap=256).
        show:       If True, show the palette.
        sns_kwargs: Additional arguments forwarded to sns.color_palette().

    Returns:
        palette:    A list of colors.

    Examples:
        color_palette("default")
        color_palette("adls")
        color_palette("viridis", alpha=0.5)
        color_palette("viridis", alpha=[1.0, 0.5])
        color_palette("default", desat=0.5)
        color_palette("default", alpha=0.5, mix_color="white")
        color_palette("default", alpha=0.5, mix_color="black")
        color_palette("default", alpha=[1.0, 0.5], grouped=True, mix_color="white")
        color_palette("viridis", show=True)
        color_palette("rocket", sns_kwargs={as_cmap=True})
        color_palette("hsl", sns_kwargs={n_colors=8})
        color_palette("default", alpha=0.5, as_cmap=True)
    """
    if name=="default":
        palette = [ "#2D8FF3", "#FC585E", "#1AAF54" ]
    elif name=="adls":
        palette = [ "#4592D5", "#2CAF9A", "#B35F5F", "#FFD700" ]
    else:
        palette = sns.color_palette(name, **sns_kwargs)
        if not isinstance(name, str):
            # Needed for ListedColormap below
            name = "custom"
        
    # Convert to RGBA, with alpha=1 if A is not set
    palette = [mplc.to_rgba(c) for c in palette]

    if desat is not None:
        if isinstance(desat, (float, int)):
            # desaturate() drops the alpha channel!
            assert not any((len(c)==4 and c[3]<1.0) for c in palette)
            palette = [sns.desaturate(c, desat) for c in palette]
        elif isinstance(desat, (list, tuple)):
            if grouped:
               palette = [sns.desaturate(c, d) for c in palette for d in desat]
            else:
                palette = [sns.desaturate(c, d) for d in desat for c in palette]

    if alpha is not None:
        if isinstance(alpha, (float, int)):
            palette = [mplc.to_rgba(c, alpha) for c in palette]
        elif isinstance(alpha, (list, tuple)):
            if grouped:
                palette = [mplc.to_rgba(c, a) for c in palette for a in alpha]
            else:
                palette = [mplc.to_rgba(c, a) for a in alpha for c in palette]

    if mix_color is not None:
        mix_color = mplc.to_rgba(mix_color)
        palette = [mix_colors_rgba(c, mix_color, mode="blend", gamma=1.0) for c in palette]

    if as_cmap and not isinstance(palette, mplc.Colormap):
        n = n_cmap
        splits = np.array_split(np.arange(n-1), len(palette))
        split_sizes = [len(s) for s in splits]
        colors = []
        for i in range(len(palette)-1):
            col_a = palette[i]
            col_b = palette[i+1]
            # Mode must be "mix" in presence of alpha
            # "mix" and "blend" are equivalent if no alpha
            colors += color_transition(col_a, col_b,
                                       mode="mix",
                                       n_steps=split_sizes[i])
        colors += [palette[-1]]
        palette = mplc.ListedColormap(colors, name)

    if show:
        if isinstance(palette, mplc.Colormap):
            colors = palette(np.linspace(0, 1, 256))
        else:
            colors = palette
        sns.palplot(colors)

    return palette


def hex_to_rgb(hex, scale=1):
    """
    Converts #9b59b6 into [155, 89, 182]
    """
    if not isinstance(hex, str) or not hex.startswith("#"):
        return None
    rgb = np.frombuffer(bytearray.fromhex(hex.lstrip("#")),
                        dtype=np.uint8)
    if scale != 1:
        rgb = rgb/scale
    return tuple(rgb)


def rgb_to_hex(rgb, scale=1):
    """
    Converts [155, 89, 182] into #9b59b6
    """
    if isinstance(rgb,(float,int)):
        rgb = np.asarray([rgb]*3)
    rgb = np.asarray(rgb)*scale
    rgb = rgb.round().astype(int).tolist()
    return '#'+bytes(rgb).hex()


def mix_colors_rgba(color_a, color_b, mode="mix", 
                    t=None, gamma=2.2, alpha=None):
    """
    Mix two colors color_a and color_b.

    Arguments:
        color_a:    Real-valued 4-tuple. Foreground color in "blend" mode.
        color_b:    Real-valued 4-tuple. Background color in "blend" mode.
        mode:       "mix":   Interpolate between two colors.
                    "blend": Blend two translucent colors.
        t:          Mixing threshold.
        gamma:      Parameter to control the gamma correction.
        alpha:      Fixed alpha value. If None, the alpha value is computed 
                    according to mixing / blending rules.

    Returns:
        rgba:       A 4-tuple with the result color.

    Source / ideas: 
        https://stackoverflow.com/questions/726549
        
        To reproduce Markus Jarderot's solution:
                mix_colors_rgba(a, b, mode="blend", t=0, gamma=1.)
        To reproduce Fordi's solution:
                mix_colors_rgba(a, b, mode="mix", t=t, gamma=2.)
        To compute the RGB color of a translucent color on white background:
                mix_colors_rgba(a, [1,1,1,1], mode="blend", t=0, gamma=None)
    """
    assert(mode in ("mix", "blend"))
    assert(gamma is None or gamma>0)
    if isinstance(color_a, str):
        color_a = mplc.to_rgba(color_a)
    if isinstance(color_b, str):
        color_b = mplc.to_rgba(color_b)
    t = t if t is not None else (0.5 if mode=="mix" else 0.)
    t = max(0,min(t,1))
    color_a = np.asarray(color_a)
    color_b = np.asarray(color_b)
    if mode=="mix" and gamma in (1., None):
        r, g, b, a = (1-t)*color_a + t*color_b
    elif mode=="mix" and gamma > 0:
        r,g,b,_ = np.power((1-t)*color_a**gamma + t*color_b**gamma, 1/gamma)
        a = (1-t)*color_a[-1] + t*color_b[-1]
    elif mode=="blend":
        alpha_a = color_a[-1]*(1-t)
        a = 1 - (1-alpha_a) * (1-color_b[-1])
        s = color_b[-1]*(1-alpha_a)/a
        if gamma in (1., None):
            r, g, b, _ = (1-s)*color_a + s*color_b
        elif gamma > 0:
            r, g, b, _ = np.power((1-s)*color_a**gamma + s*color_b**gamma,
                                  1/gamma)
    if alpha is not None:
        a = alpha
    return tuple(np.clip([r,g,b,a], 0, 1))


def color_transition(color_a, color_b, mode="mix", gamma=None,
                     n_steps=100, func=None):
    if isinstance(color_a, str):
        color_a = hex_to_rgb(color_a, scale=255)
    if isinstance(color_b, str):
        color_b = hex_to_rgb(color_b, scale=255)

    if len(color_a)==3:
        color_a = list(color_a) + [1.0]
    if len(color_b)==3:
        color_b = list(color_b) + [1.0]

    ts = np.linspace(0, 1, n_steps)
    if func is not None:
        ts = func(ts)
    palette = [mix_colors_rgba(color_a, color_b, t=t, mode=mode, gamma=gamma)
               for t in ts]
    return palette


def color_transitions(*colors, **kwargs):
    """
    Creates a color palette by interpolating between a list of colors.

    Arguments:
        *colors:    A list of colors.
        mode:       "mix":   Interpolate between two colors.
                    "blend": Blend two translucent colors.
        gamma:      Parameter to control the gamma correction.
        n_steps:    Number of steps in the interpolation.

    Returns:
        palette:    A list of colors.
    """
    n_transitions = len(colors)-1
    n_steps = kwargs.pop("n_steps", 100)
    
    # Split n_steps into n_transitions parts of (almost) equal size
    q, r = divmod(n_steps+1, n_transitions)
    n_steps_arr = np.full(n_transitions, q, dtype=int)
    n_steps_arr[:r] += 1

    palette = []
    for i in range(n_transitions):
        part = color_transition(colors[i], colors[i+1], 
                                n_steps=n_steps_arr[i], 
                                **kwargs)
        palette += part[:-1] if (i<n_transitions-1) else part
    return palette


def colors2plotly(colors, alpha=None):
    """
    Converts a list of colors into a list of plotly colors.
    """
    def c2pRGBA(c, a=None):
        if a is None and len(c)==3:
            return f"rgb{tuple(c[:3])}"
        elif a is None and len(c)==4:
            return f"rgba{tuple(c)}"
        else:
            return f"rgba{tuple(c[:3]+(a,))}"

    return [c2pRGBA(c, a=alpha) for c in colors] 

