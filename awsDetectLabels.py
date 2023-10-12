import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible


class AWSDetectLabels(AbstractFunction, GPUCompatible):
    @property
    def name(self) -> str:
        return "awsdetectlabels"

    @setup(cacheable=True, function_type="awsRekognition", batchable=True)
    def setup(self):
        print("Hello from AWSDetectLabels")
        self.device = "cpu"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            ),
        ],
        output_signatures=[
            PandasDataframe(
                columns=[
                    "name",
                    "confidence",
                ],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[
                    (None,),
                    (None,),
                ],
            ),
        ],
    )
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        import boto3
        import cv2

        client = boto3.client("rekognition")
        _, b = cv2.imencode(".jpg", frames.iloc[0].iloc[0])
        b = b.tobytes()
        response = client.detect_labels(Image={"Bytes": b}, MaxLabels=10)

        tmp = {
            "name": [],
            "confidence": [],
        }

        for label in response["Labels"]:
            tmp["name"].append(label["Name"])
            tmp["confidence"].append(label["Confidence"])

        outcome = [tmp]

        # return frames
        return pd.DataFrame(
            outcome,
            columns=["name", "confidence"],
        )

    def to_device(self, device: str):
        self.device = device
        return self
