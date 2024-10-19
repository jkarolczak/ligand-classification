import os

import streamlit as st

from deploy.inference import predict, load_model, ligand_dict, render_table, raw_pred_to_top10_dataframe
from deploy.parsing import parse_streamlit
from deploy.preprocessing import preprocess, scale_cryoem_blob
from deploy.visualization import volume_3d

model = None


def main():
    st.set_page_config(
        page_title="Ligand classification",
        page_icon="⚛",
        layout="wide"
    )

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    st.markdown("""
    # Ligand classification using deep learning
    
    ---
    By using this application, you can run an inference pipeline that takes 3D point clouds representing ligands as 
    inputs and outputs a list of the 10 most likely ligands based on the inputs. The classification is made using deep 
    learning model for place recognition problem, namely [MinkLoc3DV2](https://arxiv.org/abs/2203.00972). Model weights 
    fitted to solve the problem of ligand classification
    can be downloaded from [here (115MB)](https://github.com/jkarolczak/ligand-classification/raw/main/model.pt). To run 
    inference, upload a file using input field on the left-hand-side. On the left panel, you will see a visualization 
    of the uploaded ligand and on the right panel, you will see a ranking of the 10 most likely classes that the ligand 
    belongs to. More information about ligands can be found in the table. All ligands within a class can be viewed by 
    clicking on the class name. Additionally there is list of building ligands that are visualised when there is a 
    cursor on the  name. Click on a ligand's name to see its Protein Data Bank entry.
    
    This application serves as supplementary material for the article 
    *Deep Learning Methods for Ligand Identification in Density Maps* by Jacek Karolczak, Anna Przybyłowska, Konrad Szewczyk, 
    Witold Taisner, John M. Heumann, Michael H.B. Stowell, Michał Nowicki and Dariusz Brzezinski. The source code is publicly 
    available on [GitHub](https://github.com/jkarolczak/ligand-classification)
    
    --- 
    """)

    st.markdown("""
    ## Input examples
    You can test the application using the examples below.
    """)

    cols = st.columns(7)
    for col, file in zip(cols, os.listdir("src/deploy/ligands")):
        file_path = os.path.join("src/deploy/ligands", file)
        with open(file_path, "rb") as fp:
            col.download_button(
                label=file,
                data=fp.read(),
                file_name=file
            )

    global model
    model = load_model()

    col1, col2 = st.columns(2)
    col1.markdown("## Input")
    col1_form = col1.form("test")
    col1_content = col1.empty()
    with col1_content.container():
        col1_content.markdown("""
        Files of the following structures are supported:
        - `.npy`, `.npz`:
            - dense three dimensional numpy array
        - `.ccp4`, `.mrc`, `.map`
            - the map must be resampled to have a grid with 0.2 Å resolution
            - for details see [ccp-em website](https://www.ccpem.ac.uk/mrc_format/mrc2014.php)
        - `.xyz`, `.txt`:
            - without any header
            - each line describe a voxel following the pattern `x y z density`
        - `.pts`
            - the first line contains information about number of points (lines)
            - each line describe a voxel following the pattern `x y z density`
        - `.csv`
            - files with headers and headerless are supported
            - each line describe a voxel following the pattern `x, y, z, density`
        """)
    col2.markdown("## Predictions")
    col2_predictions = col2.empty()
    col2_predictions.info("Upload a blob to see predictions.")
    with col1_form:
        file_val = st.file_uploader("Input",
                                    type=["npy", "npz", "mrc", "ccp4", "map", "ply", "pts", "xyz", "txt", "csv"])
        form_col1, form_col2 = col1_form.columns(2)
        with form_col1:
            rescale_cryoem = st.toggle("Rescale voxel values (turn this on for cryoEM blobs)")

        with form_col2:
            resolution = st.slider("Resolution (used only for cryoEM blobs)", min_value=1.0, max_value=4.0, value=1.0,
                                   step=0.1)

        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if file_val:
                blob = parse_streamlit(file_val)
                if rescale_cryoem:
                    blob = scale_cryoem_blob(blob, resolution=resolution)
                viz = volume_3d(blob, "Blob")
                col1_content.plotly_chart(viz, use_container_width=True, height=500)
                blob = preprocess(blob)
                pred = predict(blob, model)
                df = raw_pred_to_top10_dataframe(pred)
                preds = render_table([(i + 1, ligand_dict()[row["Class"]], round(row["Probability"], 2))
                                      for i, row in df.iterrows()])
                with col2_predictions.container():
                    st.components.v1.html(preds, width=650, height=1000, scrolling=True)
            else:
                col2_predictions.error("No file has been uploaded.")


if __name__ == "__main__":
    main()
