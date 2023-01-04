from skimage.measure import marching_cubes
import plotly.figure_factory as ff


def volume_3d(blob, title, cutoff_val=0.0, grid_unit=0.2, color="#009988", opacity=0.5, save=False):
    """
    Creates an interactive #d visualization of a given volume.
    :param blob: 3D numpy array
    :param title: title of plot
    :param cutoff_val: value considered as void
    :param grid_unit: unit of each voxel; by default 0.2 Angstrom
    :param color: mesh color
    :param opacity: mesh opacity
    :param save: if True than image is saved to a file
    :return: plotly figure
    """
    verts, faces, _, _ = marching_cubes(
        blob, cutoff_val, spacing=(grid_unit, grid_unit, grid_unit)
    )
    fig = ff.create_trisurf(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        title=" ",
        simplices=faces,
        colormap=[color, color],
        show_colorbar=False,
    )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=700, )
    if opacity < 1:
        fig["data"][0].update(opacity=opacity)

    if save:
        fig.write_image(f"{title}.png")
    else:
        return fig
