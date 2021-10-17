
def get_ligand_diagram_url(ligand_id: str):
    """
    Gets RCSB's ligand diagram in SVG format. The function can be used to display the diagram in a jupyter notebook or
    in a dashboard.

    Example: plotting.get_ligand_diagram_url("SO4")
    :param ligand_id: Three-letter ligand identifier.
    :return: the url of the diagram
    """
    return f"https://cdn.rcsb.org/images/ccd/unlabeled/{ligand_id[0]}/{ligand_id}.svg"


def svg_to_html(svg_url, width="300px"):
    """
    Creates an html image of a svg, that can have its size adjusted
    Example: plotting.svg_to_html(plotting.get_ligand_diagram_url("SO4"))
    :param svg_url: url of the SVG image
    :param width: width of the output html img; can be set in % o px units
    :return: an IPython HTML object that can be displayed in a Jupyter notebook
    """
    from IPython.display import SVG, display, HTML
    import base64
    _html_template='<img width="{}" src="{}" >'

    text = _html_template.format(width, svg_url)
    return HTML(text)


def plot_interactive_trisurf(blob, title, cutoff_val=0.0, grid_unit=0.2, color='#009988', opacity=0.5):
    """
    Creates an interactive #d visualization of a given volume.
    :param blob: 3D numpy array
    :param title: title of plot
    :param cutoff_val: value considered as void
    :param grid_unit: unit of each voxel; by default 0.2 Angstrom
    :param color: mesh color
    :param opacity: mesh opacity
    :return: plotly figure
    """
    from skimage.measure import marching_cubes_lewiner
    import plotly.figure_factory as ff

    verts, faces, _, _ = marching_cubes_lewiner(blob, cutoff_val, spacing=(grid_unit, grid_unit, grid_unit))
    fig = ff.create_trisurf(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], title=title,
                            simplices=faces, colormap=[color, color], show_colorbar=False)
    if opacity < 1:
        fig['data'][0].update(opacity=opacity)

    fig.show()
