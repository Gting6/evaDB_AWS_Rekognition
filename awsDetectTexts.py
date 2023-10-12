import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible


class AWSDetectTexts(AbstractFunction, GPUCompatible):
    @property
    def name(self) -> str:
        return "awsdetecttexts"

    @setup(cacheable=True, function_type="awsRekognition", batchable=True)
    def setup(self):
        print("Hello from AWSDetectTexts")
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
                    "texts",
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
        response = client.detect_text(Image={"Bytes": b})

        tmp = {
            "texts": [],
            "confidence": [],
        }

        textDetections = response["TextDetections"]

        for text in textDetections:
            tmp["texts"].append(text["DetectedText"])
            tmp["confidence"].append(text["Confidence"])

        outcome = [tmp]

        # return frames
        return pd.DataFrame(
            outcome,
            columns=[
                "texts",
                "confidence",
            ],
        )

    def to_device(self, device: str):
        self.device = device
        return self
