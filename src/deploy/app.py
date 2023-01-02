import streamlit as st

from deploy.inference import predict, load_model
from deploy.parsing import parse
from deploy.preprocessing import preprocess
from deploy.visualization import volume_3d


model = None


def main():
    st.set_page_config(
        page_title="Ligand classification",
        page_icon="⚛",
        layout="wide"
    )

    st.markdown("""
    # Ligand classification using deep learning
    
    ---
    By using this application, you can run an inference pipeline that takes 3D point clouds representing ligands as 
    inputs and outputs a list of the 10 most likely ligands based on the inputs. To run inference, upload a file using 
    input field on the right-hand-side. On the left panel, you will see a visualization of the uploaded ligand and on 
    the right panel, you will see a ranking of the 10 most likely classes that the ligand belongs to. More information 
    about ligands can be found in the table. All ligands within a class can be viewed by clicking on the class name. 
    Additionally there is list of building ligands that are visualised when there is a cursor on the  name. Click on a 
    ligand's name to see its Protein Data Bank entry.
    
    The application was created as a Bachelor dissertation by Anna Przybyłowska, Konrad Szewczyk, Witold Taisner and 
    Jacek Karolczak, under supervision of Ph.D. Dariusz Brzeziński.
    
    --- 
    """)

    global model
    model = load_model()

    col1, col2 = st.columns(2)
    col1.markdown("## Input")
    col2.markdown("## Predictions")
    col2_predictions = col2.empty()
    col2_predictions.info("Upload a blob to see predictions.")
    with col1.form("test"):
        file_val = st.file_uploader("Input", type=["npy", "npz"])
        if st.form_submit_button():
            if file_val:
                blob = parse(file_val)
                viz = volume_3d(blob, "Blob")
                col1.plotly_chart(viz, use_container_width=True, height=500)
                blob = preprocess(blob)
                preds = predict(blob, model)
                with col2_predictions.container():
                    st.components.v1.html(preds, width=650, height=1000, scrolling=True)
            else:
                col2_predictions.error("No file has been uploaded.")


if __name__ == "__main__":
    main()
