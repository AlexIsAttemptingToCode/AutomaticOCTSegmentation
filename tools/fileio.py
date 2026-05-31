from pathlib import Path
import pandas as pd
import json

def ensure_dir(path):
    """Create a dir recursively if it doesn't exist. 
    Returns True if the folder exists or was created."""
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    return path.is_dir()


def load_audio(path, ensure1d=False, **kwargs):
    """Load an audio file from the given path. 
    Returns a tuple (audio, sample_rate)"""
    import soundfile as sf
    kwargs.setdefault("dtype", "float32")
    kwargs.setdefault("always_2d", True)
    x, fs = sf.read(path, **kwargs)
    if ensure1d and x.ndim > 1:
        x = x[:, 0]
    return x.T, fs


def count_words_in_notebook(notebook_path):
    def count_words(dct, cell_type="markdown"):
        word_count = 0
        for each in data['cells']:
            if each['cell_type'] == cell_type:
                content = each['source']
                for line in content:
                    temp = [word for word in line.split() if "#" not in word]
                    word_count += len(temp)
        return word_count

    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        n_markdown = count_words(data, "markdown")
        n_code = count_words(data, "code")
        data = {}
        data["Markdown"] = n_markdown
        data["Code"] = n_code
        return pd.DataFrame({"Counts": data})


def count_words_in_current_notebook():
    import ipynbname
    notebook_path = ipynbname.path()
    return count_words_in_notebook(notebook_path)


def check_word_counts(limits=dict(markdown=3000, code=2000)):

    limits = {"Markdown": limits["markdown"],
              "Code": limits["code"]}
    counts = count_words_in_current_notebook()

    okay_str = "✅ OK"
    fail_str = "❌ FAIL"

    okay_fgcolor = "#30a030"
    fail_fgcolor = "#f03030"
    okay_bgcolor = "#80f08033"
    fail_bgcolor = "#f0808033"
    font_weight = "bold"

    cell_fmt = 'background-color: {}; color: {};'
    cell_fmt += 'font-weight: bold; text-align: left'

    counts["Limits"] = pd.Series(limits)
    counts["Check"] = counts["Counts"] > pd.Series(limits)
    counts["Check"] = counts["Check"].map({True: fail_str,
                                           False: okay_str})

    # Formatting...
    def highlight_rows(s):
        if s["Check"] == fail_str:  # if check is True
            return [cell_fmt.format(fail_bgcolor, fail_fgcolor)] * len(s)
        else:
            return [cell_fmt.format(okay_bgcolor, okay_fgcolor)] * len(s)

    counts.index.name = "Cell type"
    counts = counts.reset_index()
    counts = counts.style.apply(highlight_rows, axis=1)
    counts = counts.set_table_styles(
          [{
              'selector': 'th.col_heading',   # column headers
              'props': [('text-align', 'left'),
                        ('font-size', '100%'),
                        ('font-weight', 'bold'),
                        ]
          }, {
              'selector': 'thead tr',         # header row
              'props': [('border-bottom', '1px solid darkgrey')]  # the ruler
          }
        ])
    counts = counts.hide(axis="index")


    # Display
    from .plotting import show_header
    show_header("Current notebook cell counts", alignment="left")
    display(counts)
