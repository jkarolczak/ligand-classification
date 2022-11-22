import os
import tarfile
from typing import List, Tuple

from azure.storage.blob import BlobServiceClient


# before running this script:
# pip install azure-storage-blob azure-identity
# export AZURE_STORAGE_CONNECTION_STRING="<ping-jkarolczak-for-the-value-to-put-here>"


def get_azure_connection() -> BlobServiceClient:
    """
    Creates connection to the Azure Data Storage.
    """
    connect_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    return blob_service_client


def make_tarfile(output_filename: str, source_dir: str) -> Tuple[str, str]:
    """
    Create a directory `/tmp/ligand` and create a tarball with compressed dataset.
    """
    os.makedirs("/tmp/ligand", exist_ok=True)
    output_filename = output_filename + ".tar.gz"
    output_path = os.path.join("/tmp/ligand", output_filename)
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    return output_filename, output_path


def list_datasets_dir(root_dir: str = "../data") -> List[Tuple[str, str]]:
    """
    List directories that stores transformed ligand dataset.
    """
    names = os.listdir(root_dir)
    return [
        (name, os.path.join(root_dir, name)) for name in names if os.path.isdir(os.path.join(root_dir, name))
    ]


if __name__ == "__main__":
    azure_conn = get_azure_connection()
    ligand_ds = [list_datasets_dir()[-3]]
    ligand_ds = list_datasets_dir()
    for ds_name, ds_path in ligand_ds:
        out_file, out_path = make_tarfile(ds_name, ds_path)
        blob_client = azure_conn.get_blob_client(container="ligands", blob=f"{ds_name}.tar.gz")
        with open(out_path, mode="rb") as fp:
            blob_client.upload_blob(fp, overwrite=True)
