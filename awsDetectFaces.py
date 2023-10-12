import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.gpu_compatible import GPUCompatible


class AWSDetectFaces(AbstractFunction, GPUCompatible):
    @property
    def name(self) -> str:
        return "awsdetectfaces"

    @setup(cacheable=True, function_type="awsRekognition", batchable=True)
    def setup(self):
        print("Hello from AWSDetectFaces")
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
                    "age",
                    "gender",
                    "smile",
                    "eyeglasses",
                    "occluded",
                    "emotion",
                ],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.STR,
                    NdArrayType.BOOL,
                    NdArrayType.BOOL,
                    NdArrayType.BOOL,
                    NdArrayType.STR,
                ],
                column_shapes=[
                    (None,),
                    (None,),
                    (None,),
                    (None,),
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
        response = client.detect_faces(Image={"Bytes": b}, Attributes=["ALL"])

        tmp = {
            "age": [],
            "gender": [],
            "smile": [],
            "eyeglasses": [],
            "occluded": [],
            "emotion": [],
        }

        for faceDetail in response["FaceDetails"]:
            tmp["age"].append(
                str(faceDetail["AgeRange"]["Low"])
                + " to "
                + str(faceDetail["AgeRange"]["High"])
                + " years old"
            )

            tmp["gender"].append(str(faceDetail["Gender"]["Value"]))
            tmp["smile"].append(faceDetail["Smile"]["Value"])
            tmp["eyeglasses"].append(faceDetail["Eyeglasses"]["Value"])
            tmp["occluded"].append(faceDetail["FaceOccluded"]["Value"])
            tmp["emotion"].append(str(faceDetail["Emotions"][0]["Type"]))

        outcome = [tmp]

        # return frames
        return pd.DataFrame(
            outcome,
            columns=[
                "age",
                "gender",
                "smile",
                "eyeglasses",
                "occluded",
                "emotion",
            ],
        )

    def to_device(self, device: str):
        self.device = device
        return self
