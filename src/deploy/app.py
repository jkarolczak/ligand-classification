import streamlit as st
import yaml

from deploy.inference import predict
from deploy.parsing import parse
from deploy.visualization import volume_3d


def main():
    st.set_page_config(
        page_title="Blobaversum",
        page_icon="🏀",
        layout="wide"
    )

    st.markdown('# Ligand classification using deep learning')
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
                preds = predict(blob)
                with col2_predictions.container():
                    st.components.v1.html(preds, width=650, height=1000, scrolling=True)
            else:
                col2_predictions.error("No file has been uploaded.")


if __name__ == "__main__":
    main()
