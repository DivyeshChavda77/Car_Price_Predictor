from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":

    ingestion = DataIngestion()

    train_data, test_data = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()

    train_arr, test_arr, _ = transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    print(train_arr.shape)
    print(test_arr.shape)