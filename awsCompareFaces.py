import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible


class AWSCompareFaces(AbstractFunction, GPUCompatible):
    @property
    def name(self) -> str:
        return "awscomparefaces"

    @setup(cacheable=True, function_type="awsRekognition", batchable=True)
    def setup(self, threshold=0):
        print("Hello from AWSCompareFaces")
        self.threshold = threshold
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
                columns=["similarity"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None,)],
            ),
        ],
    )
    def forward(self, frames: pd.DataFrame) -> pd.DataFrame:
        import boto3
        import cv2

        tmp = {"similarity": []}

        _, a = cv2.imencode(".jpg", frames.iloc[0].iloc[0])
        a = a.tobytes()
        _, b = cv2.imencode(".jpg", frames.iloc[0].iloc[1])
        b = b.tobytes()
        client = boto3.client("rekognition")
        response = client.compare_faces(
            SourceImage={"Bytes": a},
            TargetImage={"Bytes": b},
            SimilarityThreshold=0,
        )
        tmp["similarity"].append(response["FaceMatches"][0]["Similarity"])

        outcome = [tmp]

        # return frames
        return pd.DataFrame(outcome, columns=["similarity"])

    def to_device(self, device: str):
        self.device = device
        return self
