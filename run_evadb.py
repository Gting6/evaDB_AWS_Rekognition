# Import the EvaDB package
import evadb
import warnings


cursor = evadb.connect().cursor()
warnings.filterwarnings("ignore")

cursor.query("DROP TABLE IF EXISTS faceDemo").df()
cursor.query("DROP TABLE IF EXISTS objectImgs").df()
cursor.query("DROP TABLE IF EXISTS textImgs").df()

cursor.query("LOAD IMAGE 'chen.jpg' INTO faceDemo").df()
cursor.query("LOAD IMAGE 'wen.jpg' INTO faceDemo").df()
cursor.query("LOAD IMAGE 'liu1.jpg' INTO faceDemo").df()
cursor.query("LOAD IMAGE 'liu2.jpg' INTO faceDemo").df()
cursor.query("LOAD IMAGE 'liu3.jpg' INTO faceDemo").df()
cursor.query("LOAD IMAGE 'liuwen.jpg' INTO faceDemo").df()

cursor.query("LOAD IMAGE 'dog.jpg' INTO objectImgs").df()

cursor.query("LOAD IMAGE 'text.jpg' INTO textImgs").df()

cursor.query(
    """
    CREATE FUNCTION IF NOT EXISTS AWSCompareFaces
    IMPL './evadb/functions/awsCompareFaces.py';
    """
).df()

cursor.query(
    """
    CREATE FUNCTION IF NOT EXISTS AWSDetectFaces
    IMPL './evadb/functions/awsDetectFaces.py';
    """
).df()

cursor.query(
    """
    CREATE FUNCTION IF NOT EXISTS AWSDetectLabels
    IMPL './evadb/functions/awsDetectLabels.py';
    """
).df()

cursor.query(
    """
    CREATE FUNCTION IF NOT EXISTS AWSDetectTexts
    IMPL './evadb/functions/awsDetectTexts.py';
    """
).df()

query = cursor.query(
    f"""
    SELECT name, AWSCompareFaces(data, Open('./liu1.jpg'))
    FROM faceDemo
    WHERE AWSCompareFaces(data, Open('./liu1.jpg')) > [80.0];
    """
)
print(query.df())

query = cursor.query(
    f"""
    SELECT name, AWSCompareFaces(data, Open('./liuwen.jpg'))
    FROM faceDemo
    WHERE AWSCompareFaces(data, Open('./liuwen.jpg')) > [80.0]
    ORDER BY AWSCompareFaces(data, Open('./liuwen.jpg'));
    """
)
print(query.df())

query = cursor.query(
    f"""
    SELECT name, AWSDetectFaces(data)
    FROM faceDemo
    """
)
print(query.df())

query = cursor.query(
    f"""
    SELECT name, AWSDetectLabels(data)
    FROM objectImgs
    """
)
print(query.df())

query = cursor.query(
    f"""
    SELECT name, AWSDetectTexts(data)
    FROM textImgs
    """
)
print(query.df())

cursor.query("DROP FUNCTION IF EXISTS AWSCompareFaces").df()
cursor.query("DROP FUNCTION IF EXISTS AWSDetectFaces").df()
cursor.query("DROP FUNCTION IF EXISTS AWSDetectLabels").df()
cursor.query("DROP FUNCTION IF EXISTS AWSDetectTexts").df()
